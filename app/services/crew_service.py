import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.core.config import SIX_ELEMENT_ORDER, load_runtime_config
from app.core.exceptions import BrainstormExecutionError
from app.rag.tool import build_sales_knowledge_tool
from app.services.parser import parse_markdown_payload
from app.services.rules import SixElementRule, format_rules_context, load_six_element_rules
from app.services.tools import build_core_sales_tools


logger = logging.getLogger(__name__)


AMBIGUOUS_DEADLINE_TERMS = (
    "尽快",
    "后续",
    "持续跟进",
    "持续关注",
    "回头",
    "抽空",
    "有空",
    "之后再说",
    "找时间",
    "择机",
)
PLACEHOLDER_VALUES = {
    "",
    "...",
    "……",
    "待补充",
    "待确认",
    "待定",
    "暂无",
    "略",
    "无",
    "n/a",
    "na",
}
INCREMENTAL_OUTPUT_PATTERNS = (
    r"动作[五六七八九十]",
    r"补充一条",
    r"补一条",
    r"补动作",
    r"承接上文",
    r"前面已给出",
    r"前文已给出",
    r"这里补",
    r"上面已经说过",
)
BLIND_SPOT_LABELS = ("核心盲区", "策略盲区", "盲区")
NEGATIVE_BLIND_SPOT_HINTS = ("无", "暂无", "未发现", "没有", "不明显", "未见", "较少")


@dataclass
class BrainstormResult:
    project_name: str
    closer_result: str
    parsed_inputs: dict[str, str]
    intermediate_results: dict[str, str] | None = None
    closer_failure_reason: str | None = None


@dataclass(frozen=True)
class TaskPromptSpec:
    description: str
    expected_output: str


@dataclass(frozen=True)
class CloserValidationResult:
    is_valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class CloserExecutionResult:
    rendered_result: str
    raw_result: str
    failure_reason: str | None = None
    structured_output: "CloserStructuredOutput | None" = None
    was_repaired: bool = False


class CloserActionItem(BaseModel):
    priority: int = Field(..., description="动作优先级，越小越优先。")
    action: str = Field(..., description="销售下一步要执行的具体动作。")
    deadline: str = Field(..., description="明确时间期限。")
    target: str = Field(..., description="动作对象。")
    goal: str = Field(..., description="动作要验证或推进的具体目标。")
    is_probe: bool = Field(default=False, description="是否为探雷/验证信息动作。")


class CloserStructuredOutput(BaseModel):
    diagnosis_summary: list[str] = Field(
        ...,
        description="简短诊断结论，2-5 条，聚焦最薄弱要素与最危险卡点。",
    )
    action_recommendations: list[CloserActionItem] = Field(
        ...,
        description="3-5 条按优先级排序的动作建议。",
    )


OUTPUT_STYLE_GUARDRAILS = (
    "统一输出风格约束：\n"
    "1. 禁用学术体、培训体、大纲体和术语直出，不要写成内部培训材料。\n"
    "2. 禁止直接输出“直接法”“要素一”“方法论”“框架”等标签。\n"
    "3. 六要素规则库和 RAG 只能作为你的内部判断依据，要隐藏背后的理论框架，对外必须翻译成销售日常工作口语。\n"
    "4. 结论必须贴近真实推进场景，直接落到下一步动作、判断或沟通方式，不要讲理论。"
)

ACTION_STYLE_REQUIREMENTS = (
    "动作表达约束：\n"
    "1. 每条动作都必须写明 动作 / 时间期限 / 对象 / 目的。\n"
    "2. 时间期限必须明确到最晚时间、阶段窗口或截止时点，不能只写“尽快”或“后续”。\n"
    "3. 目的必须写清这条动作要验证什么、推进什么或锁定什么，不能只写空泛态度。\n"
    "4. 在给出动作前，先判断这条动作的生硬度，明确它是否像审问、施压或越级。\n"
    "5. 只要动作偏硬，就必须补一条“柔和切入点”，先给出自然由头，再进入正题。"
)

CHALLENGE_STYLE_REQUIREMENTS = (
    "社交温度审查要求：\n"
    "1. 逐条指出原动作里哪些地方太硬、太直、太像审问或压人。\n"
    "2. 每条挑战意见都要给出柔和切入点和替代表达。\n"
    "3. 如果某条动作涉及敏感推进，例如拍板、预算、绩效、关系深度，优先给出顺势切入的理由。"
)


def format_project_context(project_name: str, six_element_inputs: dict[str, str]) -> str:
    """Render project inputs into a stable prompt block."""
    blocks = [f"项目名称：{project_name}"]
    for name in SIX_ELEMENT_ORDER:
        blocks.append(f"【{name}】\n{six_element_inputs.get(name, '暂无输入（待补充）')}")
    return "\n\n".join(blocks)


def build_task_prompt_specs() -> dict[str, TaskPromptSpec]:
    return {
        "diagnosis": TaskPromptSpec(
            description=(
                "请基于以下项目现状和六要素规则，做一份高质量诊断报告。\n\n"
                "【项目现状】\n{project_context}\n\n"
                "【六要素规则】\n{rules_context}\n\n"
                f"{OUTPUT_STYLE_GUARDRAILS}\n\n"
                "要求：\n"
                "0. 先调用 check_rule_violations 工具，检查当前项目是否触碰六要素红线；若工具返回无明显违规，也要明确说明。\n"
                "1. 必须按六要素逐项输出。\n"
                "2. 每个要素固定包含：已知事实、关键缺口、核心盲区、核心不足、风险等级、高危卡点。\n"
                "3. 核心盲区指的是：团队当前以为知道、但实际未经验证的关键信息，不要把猜测写成事实。\n"
                "4. 核心不足指的是：我们在产品能力、客情关系或流程卡位上存在的最大客观劣势。\n"
                "5. 只做诊断，不允许提出策略、建议、应对措施，也不要偷渡动作话术。\n"
                "6. 结合权重，明确当前最薄弱的要素和最危险的卡点。\n"
                "7. 允许用六要素做内部判断，但对外禁止显式暴露方法论框架。"
            ),
            expected_output=(
                "一份口语化的诊断报告，按六要素说明已知事实、关键缺口、核心盲区、核心不足、"
                "风险等级和高危卡点，不包含任何对策、术语标签或方法论名称。"
            ),
        ),
        "strategy": TaskPromptSpec(
            description=(
                "请只针对上一步识别出的高危卡点，输出初版销售破局动作。\n\n"
                "你必须同时参考以下规则库：\n{rules_context}\n\n"
                f"{OUTPUT_STYLE_GUARDRAILS}\n\n"
                f"{ACTION_STYLE_REQUIREMENTS}\n\n"
                "要求：\n"
                "0. 如提供了 sales_knowledge_search 工具，先检索与当前项目最相关的 product 与 sales 知识，再设计动作。\n"
                "1. 只针对高危卡点出招，不要复述全部诊断。\n"
                "2. 每条动作必须包含：对应六要素、动作、时间期限、对象、目的、生硬度判断、柔和切入点。\n"
                "3. 动作设计必须优先回应诊断里暴露出的高危卡点、核心盲区和核心不足，避免脱节。\n"
                "4. 只能依据六要素策略库和检索结果出招，不能凭空发挥。\n"
                "5. 动作必须可执行，避免抽象口号，所有术语都要翻译成客户沟通口语。\n"
                "6. 如果检索为空，必须明确基于当前项目上下文判断，不要伪造已有案例、客户反馈或产品能力。"
            ),
            expected_output=(
                "一份初版破局动作清单。每条动作都包含对应六要素、动作、时间期限、对象、目的、"
                "生硬度判断和柔和切入点，整体口语化且不出现方法论标签。"
            ),
        ),
        "challenge": TaskPromptSpec(
            description=(
                "请站在客户的降本增效领导或强势竞争对手视角，对上一步初版策略做压力测试。\n\n"
                "你必须输出：\n"
                "1. 哪些动作太天真或时机错误\n"
                "2. 哪些动作与当前采购流程冲突\n"
                "3. 竞争对手最可能的反制方式\n"
                "4. 策略盲区揭示\n"
                "5. 执行资源不足\n"
                "6. 必须修正的点\n\n"
                f"{OUTPUT_STYLE_GUARDRAILS}\n\n"
                f"{CHALLENGE_STYLE_REQUIREMENTS}\n\n"
                "要求：\n"
                "0. 如提供了 sales_knowledge_search 工具，先检索与当前项目最相关的 product 与 sales 知识，再做压力测试。\n"
                "1. 必须逐条挑战初版动作，而不是泛泛评论。\n"
                "2. 不能生成最终方案，只能输出挑战意见和必须修正点。\n"
                "3. 挑战要结合六要素中的权重、流程阶段、竞争风险和社交温度。\n"
                "4. 每条挑战意见必须逐条对应原动作，固定包含：原动作的问题、策略盲区揭示、执行资源不足、为什么时机不对或流程不匹配、竞争对手可能如何反制、必须修正的点、柔和切入点、替代表达。\n"
                "5. 策略盲区揭示要明确指出当前动作是否忽略了客户内部政治斗争、隐藏影响者或其他利益相关者。\n"
                "6. 执行资源不足要明确指出销售在执行该动作时最可能缺乏的资源、能力、背书或协同支持。\n"
                "7. 每条挑战意见都要明确指出原动作的生硬度问题，并补一条柔和切入点和替代表达。\n"
                "8. 优先引用检索到的失败教训、产品能力边界和常见交付风险；若检索为空，必须明确这一点。"
            ),
            expected_output=(
                "一份口语化的蓝军挑战报告，逐条指出原动作的问题、策略盲区揭示、执行资源不足、"
                "生硬度风险、竞争反制、必须修正点，并给出柔和切入点和替代表达。"
            ),
        ),
        "closer": TaskPromptSpec(
            description=(
                "请旁听完前面所有历史讨论后，输出最终收口结果。\n\n"
                f"{OUTPUT_STYLE_GUARDRAILS}\n\n"
                f"{ACTION_STYLE_REQUIREMENTS}\n\n"
                "动作建议判定标准：\n"
                "1. 动作建议指销售下一步可以直接执行的行为，例如约谁沟通、问哪几个关键问题、补哪类关键信息、"
                "推动哪个节点、验证哪个决策假设。\n"
                "2. 单纯描述风险、重复已有事实、空泛表态（如加强跟进、持续关注、做好准备）都不算动作建议。\n\n"
                "要求：\n"
                "0. 每条候选动作在输出前都要先调用 action_format_validator 和 action_executability_check；"
                "请把动作临时整理成“动作: ... | 时间期限: ... | 对象: ... | 目的: ...”再调用工具；如果未通过，必须先修正后再输出最终结果。\n"
                "1. 如提供了 sales_knowledge_search 工具，先检索与当前项目最相关的 product 与 sales 知识，再收口动作。\n"
                "2. 你的任务不是只做风险诊断，而是先给出简短诊断结论，再输出销售下一步最应该执行的 3-5 条动作建议。\n"
                "3. 你必须输出完整最终版，不能假设用户已经看过 strategy/challenge，也不能写“补充动作五”“前文已给出”“承接上文”。\n"
                "4. diagnosis_summary 必须有 2-5 条，action_recommendations 必须有 3-5 条。\n"
                "5. 动作建议必须按 priority 升序表达，优先覆盖六要素中最薄弱、风险最高的环节。\n"
                "6. 若存在核心盲区，动作建议中必须至少包含 1 条 is_probe=true 的探雷/验证信息动作，用来验证信息或消除盲区。\n"
                "7. 每条动作建议都必须写清楚做什么、最晚什么时候做、找谁、想验证什么，字段固定为 priority / action / deadline / target / goal / is_probe。\n"
                "8. 动作建议中的对象必须明确是销售本人、合作伙伴、技术人员、高层、客户关键人或其他具体角色。\n"
                "9. 动作建议的时间期限必须明确，不能用“尽快”“后续”这类模糊词；动作必须具体、可执行，并已经吸收挑战阶段给出的柔和切入点。\n"
                "10. 优先使用检索到的成功打法、失败教训和产品能力边界来约束最终动作。\n"
                "11. 如果信息不足，也要基于现有信息给出最稳妥的下一步动作，不要因为信息不全而放弃建议。\n"
                "12. 只输出一个 JSON 对象，不要输出 Markdown、表格、额外总结句或方法论标签。JSON 顶层字段只能是 diagnosis_summary 和 action_recommendations。"
            ),
            expected_output=(
                "一个完整 JSON 对象，包含 diagnosis_summary 与 action_recommendations。"
                "diagnosis_summary 为 2-5 条简短诊断；"
                "action_recommendations 为 3-5 条按优先级排序的动作建议，每条都包含 "
                "priority / action / deadline / target / goal / is_probe；"
                "若存在核心盲区，至少 1 条动作的 is_probe 为 true。"
            ),
        ),
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_for_comparison(text: str) -> str:
    normalized = _normalize_text(text).lower()
    return re.sub(r"[\W_]+", "", normalized)


def _contains_incremental_trace(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(re.search(pattern, normalized) for pattern in INCREMENTAL_OUTPUT_PATTERNS)


def _is_placeholder_or_thin(text: str, *, min_length: int) -> bool:
    normalized = _normalize_text(text)
    lowered = normalized.lower()
    if not normalized:
        return True
    if lowered in PLACEHOLDER_VALUES:
        return True
    if normalized in PLACEHOLDER_VALUES:
        return True
    if "待补充" in normalized or "待确认" in normalized:
        return True
    return len(normalized) < min_length


def _contains_ambiguous_deadline(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(term in normalized for term in AMBIGUOUS_DEADLINE_TERMS)


def _is_meaningful_blind_spot_line(line: str) -> bool:
    normalized = _normalize_text(line)
    if not any(label in normalized for label in BLIND_SPOT_LABELS):
        return False

    label_tail = normalized
    for label in BLIND_SPOT_LABELS:
        if label in normalized:
            label_tail = normalized.split(label, 1)[1].lstrip("：: -")
            break

    if not label_tail:
        return False
    return not any(hint in label_tail for hint in NEGATIVE_BLIND_SPOT_HINTS)


def _requires_probe_action(*texts: str) -> bool:
    for text in texts:
        if any(_is_meaningful_blind_spot_line(line) for line in text.splitlines()):
            return True
    return False


def _looks_like_probe_action(item: CloserActionItem) -> bool:
    return item.is_probe


def _extract_raw_text(raw_payload: Any) -> str:
    if raw_payload is None:
        return ""
    if isinstance(raw_payload, str):
        return raw_payload.strip()

    content = getattr(raw_payload, "content", raw_payload)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunk.strip() for chunk in chunks if chunk and chunk.strip()).strip()
    return str(content).strip()


def _should_downgrade_system_messages(api_base: str, model_name: str) -> bool:
    normalized_base = api_base.lower()
    normalized_model = model_name.lower()
    return "minimaxi.com" in normalized_base or normalized_model.startswith("minimax")


def _stringify_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                chunks.append(text if isinstance(text, str) else str(item))
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(content)


def _downgrade_system_messages_for_minimax(messages: list[Any]) -> list[Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    converted_messages: list[Any] = []
    pending_system_messages: list[str] = []

    for message in messages:
        if isinstance(message, SystemMessage):
            pending_system_messages.append(_stringify_message_content(message.content))
            continue

        if not pending_system_messages:
            converted_messages.append(message)
            continue

        system_instruction = "系统指令（请严格遵守）：\n" + "\n\n".join(pending_system_messages)
        pending_system_messages = []

        if isinstance(message, HumanMessage):
            converted_messages.append(
                HumanMessage(
                    content=(
                        f"{system_instruction}\n\n"
                        f"用户消息：\n{_stringify_message_content(message.content)}"
                    ),
                    additional_kwargs=message.additional_kwargs,
                    response_metadata=message.response_metadata,
                    name=message.name,
                    id=message.id,
                )
            )
        else:
            converted_messages.append(HumanMessage(content=system_instruction))
            converted_messages.append(message)

    if pending_system_messages:
        converted_messages.append(
            HumanMessage(
                content="系统指令（请严格遵守）：\n" + "\n\n".join(pending_system_messages)
            )
        )

    return converted_messages


def _downgrade_crewai_system_messages_for_minimax(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted_messages: list[dict[str, Any]] = []
    pending_system_messages: list[str] = []

    for message in messages:
        if message.get("role") == "system":
            pending_system_messages.append(_stringify_message_content(message.get("content", "")))
            continue

        if not pending_system_messages:
            converted_messages.append(message)
            continue

        system_instruction = "系统指令（请严格遵守）：\n" + "\n\n".join(pending_system_messages)
        pending_system_messages = []

        if message.get("role") == "user":
            converted_message = dict(message)
            converted_message["content"] = (
                f"{system_instruction}\n\n"
                f"用户消息：\n{_stringify_message_content(message.get('content', ''))}"
            )
            converted_messages.append(converted_message)
        else:
            converted_messages.append({"role": "user", "content": system_instruction})
            converted_messages.append(message)

    if pending_system_messages:
        converted_messages.append(
            {
                "role": "user",
                "content": "系统指令（请严格遵守）：\n" + "\n\n".join(pending_system_messages),
            }
        )

    return converted_messages


def _create_minimax_compatible_crewai_openai_completion(openai_completion_cls: Any) -> Any:
    class MiniMaxCompatibleOpenAICompletion(openai_completion_cls):
        def _format_messages(self, messages: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
            formatted_messages = super()._format_messages(messages)
            return _downgrade_crewai_system_messages_for_minimax(formatted_messages)

    return MiniMaxCompatibleOpenAICompletion


def _build_crewai_llm(
    *,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
) -> Any:
    if _should_downgrade_system_messages(base_url, model):
        from crewai.llms.providers.openai.completion import OpenAICompletion

        minimax_completion_cls = _create_minimax_compatible_crewai_openai_completion(
            OpenAICompletion
        )
        return minimax_completion_cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

    from crewai import LLM

    return LLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


def _create_minimax_compatible_chat_openai(chat_openai_cls: Any) -> Any:
    class MiniMaxCompatibleChatOpenAI(chat_openai_cls):
        def _generate(
            self,
            messages: list[Any],
            stop: list[str] | None = None,
            run_manager: Any | None = None,
            **kwargs: Any,
        ) -> Any:
            return super()._generate(
                _downgrade_system_messages_for_minimax(messages),
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )

        async def _agenerate(
            self,
            messages: list[Any],
            stop: list[str] | None = None,
            run_manager: Any | None = None,
            **kwargs: Any,
        ) -> Any:
            return await super()._agenerate(
                _downgrade_system_messages_for_minimax(messages),
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )

    return MiniMaxCompatibleChatOpenAI


def _build_chat_openai(
    chat_openai_cls: Any,
    *,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
) -> Any:
    resolved_cls = (
        _create_minimax_compatible_chat_openai(chat_openai_cls)
        if _should_downgrade_system_messages(base_url, model)
        else chat_openai_cls
    )
    return resolved_cls(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


def _strip_thinking_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


def _iter_json_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    cleaned_text = _strip_thinking_blocks(text)
    if cleaned_text.startswith("{") and cleaned_text.endswith("}"):
        candidates.append(cleaned_text)

    depth = 0
    start_index: int | None = None
    in_string = False
    escaped = False

    for index, char in enumerate(cleaned_text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
            continue
        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_index is not None:
                candidates.append(cleaned_text[start_index : index + 1])
                start_index = None

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(candidate.strip() for candidate in candidates if candidate.strip()))


def _parse_closer_structured_output_from_text(text: str) -> CloserStructuredOutput | None:
    for candidate in _iter_json_object_candidates(text):
        try:
            return CloserStructuredOutput.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, TypeError, ValidationError):
            continue
    return None


def _coerce_closer_structured_output(payload: Any) -> CloserStructuredOutput | None:
    if payload is None:
        return None
    if isinstance(payload, CloserStructuredOutput):
        return payload
    try:
        if isinstance(payload, str):
            return _parse_closer_structured_output_from_text(payload)
        if isinstance(payload, BaseModel):
            return CloserStructuredOutput.model_validate(payload.model_dump())
        if isinstance(payload, dict):
            return CloserStructuredOutput.model_validate(payload)
    except ValidationError:
        return None
    return None


def _render_loose_closer_result(output: CloserStructuredOutput | None) -> str:
    if output is None:
        return ""
    try:
        return render_closer_result(output)
    except Exception:
        return ""


def _format_validation_errors(errors: tuple[str, ...]) -> str:
    return "\n".join(f"- {error}" for error in errors)


def validate_closer_output(
    output: CloserStructuredOutput,
    *,
    diagnosis_result: str = "",
    challenge_result: str = "",
) -> CloserValidationResult:
    errors: list[str] = []

    diagnosis_items = [_normalize_text(item) for item in output.diagnosis_summary]
    action_items = output.action_recommendations

    if not 2 <= len(diagnosis_items) <= 5:
        errors.append("diagnosis_summary 条数必须在 2-5 条之间。")
    if not 3 <= len(action_items) <= 5:
        errors.append("action_recommendations 条数必须在 3-5 条之间。")

    for index, diagnosis in enumerate(diagnosis_items, start=1):
        if _is_placeholder_or_thin(diagnosis, min_length=6):
            errors.append(f"第 {index} 条 diagnosis_summary 过薄或仍是占位内容。")
        if _contains_incremental_trace(diagnosis):
            errors.append(f"第 {index} 条 diagnosis_summary 出现增量式收口痕迹。")

    seen_actions: set[str] = set()
    for index, item in enumerate(action_items, start=1):
        if item.priority < 1:
            errors.append(f"第 {index} 条动作的 priority 必须是正整数。")
        if _is_placeholder_or_thin(item.action, min_length=8):
            errors.append(f"第 {index} 条动作的 action 过薄或仍是占位内容。")
        if _is_placeholder_or_thin(item.deadline, min_length=2):
            errors.append(f"第 {index} 条动作的 deadline 不能为空。")
        elif _contains_ambiguous_deadline(item.deadline):
            errors.append(f"第 {index} 条动作的 deadline 使用了模糊时间。")
        if _is_placeholder_or_thin(item.target, min_length=2):
            errors.append(f"第 {index} 条动作的 target 过薄或仍是占位内容。")
        if _is_placeholder_or_thin(item.goal, min_length=8):
            errors.append(f"第 {index} 条动作的 goal 过薄或仍是占位内容。")

        merged_text = " ".join((item.action, item.deadline, item.target, item.goal))
        if _contains_incremental_trace(merged_text):
            errors.append(f"第 {index} 条动作出现增量式收口痕迹。")

        duplicate_key = _normalize_for_comparison(f"{item.action}|{item.target}|{item.goal}")
        if duplicate_key in seen_actions:
            errors.append(f"第 {index} 条动作与前文动作重复。")
        else:
            seen_actions.add(duplicate_key)

    if _requires_probe_action(diagnosis_result, challenge_result) and not any(
        _looks_like_probe_action(item) for item in action_items
    ):
        errors.append("上游识别出盲区，但最终动作中没有 is_probe=true 的探雷/验证信息动作。")

    return CloserValidationResult(is_valid=not errors, errors=tuple(errors))


def render_closer_result(output: CloserStructuredOutput) -> str:
    diagnosis_lines = "\n".join(f"- {_normalize_text(item)}" for item in output.diagnosis_summary)
    sorted_actions = sorted(
        output.action_recommendations,
        key=lambda item: (item.priority, item.action, item.target),
    )
    action_lines = "\n".join(
        (
            f"{index}. 动作：{_normalize_text(item.action)} | 时间期限：{_normalize_text(item.deadline)} | "
            f"对象：{_normalize_text(item.target)} | 目的：{_normalize_text(item.goal)}"
        )
        for index, item in enumerate(sorted_actions, start=1)
    )
    return f"一、简短诊断\n{diagnosis_lines}\n\n二、执行动作建议\n{action_lines}"


def _invoke_structured_closer(
    llm: Any,
    prompt: str,
) -> tuple[CloserStructuredOutput | None, str, str | None]:
    try:
        runnable = llm.with_structured_output(CloserStructuredOutput, include_raw=True)
        response = runnable.invoke(prompt)
    except Exception as error:
        return None, "", str(error)

    parsed_output: CloserStructuredOutput | None = None
    raw_text = ""
    parsing_error: str | None = None

    if isinstance(response, dict):
        parsed_output = _coerce_closer_structured_output(response.get("parsed"))
        raw_text = _extract_raw_text(response.get("raw"))
        if response.get("parsing_error") is not None:
            parsing_error = str(response["parsing_error"])
    else:
        parsed_output = _coerce_closer_structured_output(response)
        raw_text = _extract_raw_text(response)

    if parsed_output is None and raw_text:
        parsed_output = _coerce_closer_structured_output(raw_text)
        if parsed_output is None:
            try:
                CloserStructuredOutput.model_validate_json(raw_text)
            except ValidationError as error:
                parsing_error = str(error)
            except ValueError:
                parsing_error = "未能从模型原文中提取合法 JSON 结构。"

    return parsed_output, raw_text, parsing_error


def repair_closer_output(
    *,
    repair_llm: Any,
    raw_closer_output: str,
    validation_errors: tuple[str, ...],
    diagnosis_result: str,
    strategy_result: str,
    challenge_result: str,
) -> tuple[CloserStructuredOutput | None, str, str | None]:
    prompt = (
        "你正在修复一份销售收口结果。请重新输出完整最终版，而不是增量补充。\n\n"
        "硬性要求：\n"
        "1. 只输出完整结构化结果，不要输出 Markdown、表格或解释。\n"
        "2. diagnosis_summary 必须有 2-5 条。\n"
        "3. action_recommendations 必须有 3-5 条。\n"
        "4. 每条动作必须包含 priority / action / deadline / target / goal / is_probe。\n"
        "5. deadline 不能写成“尽快”“后续”“持续跟进”等模糊值。\n"
        "6. 如果上游识别出盲区，至少一条动作的 is_probe 必须为 true。\n"
        "7. 禁止写“补充动作五”“前文已给出”“承接上文”等增量式痕迹。\n"
        "8. 你不能假设用户已经看过前面的 strategy/challenge，必须一次性输出完整最终版。\n\n"
        "【当前失败原因】\n"
        f"{_format_validation_errors(validation_errors)}\n\n"
        "【原始收口结果】\n"
        f"{raw_closer_output or '（空）'}\n\n"
        "【诊断阶段输出】\n"
        f"{diagnosis_result}\n\n"
        "【策略阶段输出】\n"
        f"{strategy_result}\n\n"
        "【挑战阶段输出】\n"
        f"{challenge_result}"
    )
    return _invoke_structured_closer(repair_llm, prompt)


def stabilize_closer_output(
    *,
    initial_raw_text: str,
    initial_structured_output: Any,
    diagnosis_result: str,
    strategy_result: str,
    challenge_result: str,
    repair_llm: Any,
    max_repair_attempts: int = 2,
) -> CloserExecutionResult:
    parsed_output = _coerce_closer_structured_output(
        initial_structured_output
    ) or _coerce_closer_structured_output(initial_raw_text)
    fallback_text = initial_raw_text.strip() or _render_loose_closer_result(parsed_output)
    validation_errors: tuple[str, ...]
    repaired = False

    if parsed_output is None:
        validation_errors = ("closer 未返回合法结构化结果。",)
    else:
        validation = validate_closer_output(
            parsed_output,
            diagnosis_result=diagnosis_result,
            challenge_result=challenge_result,
        )
        if validation.is_valid:
            return CloserExecutionResult(
                rendered_result=render_closer_result(parsed_output),
                raw_result=initial_raw_text.strip() or render_closer_result(parsed_output),
                structured_output=parsed_output,
            )
        validation_errors = validation.errors

    latest_failure_reason = "；".join(validation_errors)
    current_raw_text = initial_raw_text.strip()

    for _ in range(max_repair_attempts):
        repaired = True
        repaired_output, repaired_raw_text, parsing_error = repair_closer_output(
            repair_llm=repair_llm,
            raw_closer_output=current_raw_text or fallback_text,
            validation_errors=validation_errors,
            diagnosis_result=diagnosis_result,
            strategy_result=strategy_result,
            challenge_result=challenge_result,
        )

        if repaired_raw_text.strip():
            current_raw_text = repaired_raw_text.strip()
            fallback_text = current_raw_text
        elif repaired_output is not None:
            fallback_text = _render_loose_closer_result(repaired_output) or fallback_text

        if repaired_output is None:
            latest_failure_reason = parsing_error or "repair 未返回合法结构化结果。"
            validation_errors = (latest_failure_reason,)
            continue

        validation = validate_closer_output(
            repaired_output,
            diagnosis_result=diagnosis_result,
            challenge_result=challenge_result,
        )
        if validation.is_valid:
            return CloserExecutionResult(
                rendered_result=render_closer_result(repaired_output),
                raw_result=current_raw_text or render_closer_result(repaired_output),
                structured_output=repaired_output,
                was_repaired=True,
            )

        validation_errors = validation.errors
        latest_failure_reason = "；".join(validation.errors)

    final_fallback = fallback_text or _render_loose_closer_result(parsed_output)
    if not final_fallback:
        final_fallback = (
            "一、简短诊断\n"
            "- 当前收口结果未能稳定落版，请结合诊断、策略、挑战三阶段输出人工复核。\n"
            "- 优先补齐核心盲区与最危险卡点对应的下一步动作。\n\n"
            "二、执行动作建议\n"
            "1. 动作：复核上游三阶段输出并补齐缺失字段 | 时间期限：今天下班前 | 对象：销售负责人 | 目的：恢复一版可直接执行的下一步动作计划。"
        )

    logger.warning("Closer stabilization fell back to semi-finished text: %s", latest_failure_reason)
    return CloserExecutionResult(
        rendered_result=final_fallback,
        raw_result=current_raw_text or final_fallback,
        failure_reason=latest_failure_reason,
        structured_output=parsed_output,
        was_repaired=repaired,
    )


def build_sales_battle_crew(
    shared_llm: Any,
    debug_mode: bool = False,
    knowledge_tool: Any | None = None,
    closer_llm: Any | None = None,
):
    """Build the six-element battle crew with a structured closer."""
    try:
        from crewai import Agent, Crew, Process, Task
    except ModuleNotFoundError as error:
        raise BrainstormExecutionError(
            "缺少 crewai。请先激活虚拟环境后安装依赖：pip install crewai"
        ) from error

    prompt_specs = build_task_prompt_specs()
    core_tools = build_core_sales_tools()
    diagnostician_tool = core_tools["check_rule_violations"]
    action_format_tool = core_tools["action_format_validator"]
    action_executability_tool = core_tools["action_executability_check"]

    diagnostician_agent = Agent(
        role="情报侦察兵（The Diagnostician）",
        goal="严格对照项目六要素标准，找出当前项目情报缺口、风险、高危卡点、伪确定信息和客观劣势。",
        backstory=(
            "你是专门挑刺的销售情报侦察兵。你不负责想对策，只负责对照标准找盲区、"
            "识别那些以为知道但实际未经验证的伪确定信息，揭露我们在产品能力、客情关系"
            "和流程卡位上的客观劣势，并按风险高低排序。"
        ),
        llm=shared_llm,
        verbose=debug_mode,
        allow_delegation=False,
        use_system_prompt=False,
        tools=[diagnostician_tool],
    )

    strategist_agent = Agent(
        role="破局军师（The Strategist）",
        goal="针对高危卡点，严格依据六要素策略库设计初版销售破局动作。",
        backstory=(
            "你是销售破局军师，擅长把项目弱点转化为下一步推进动作。"
            "你不能空谈，必须优先回应诊断暴露出的高危卡点、核心盲区和核心不足，"
            "把动作写清楚：目标、执行动作、依赖资源、触达人和预期结果。"
        ),
        llm=shared_llm,
        verbose=debug_mode,
        allow_delegation=False,
        use_system_prompt=False,
        tools=[knowledge_tool] if knowledge_tool is not None else [],
    )

    challenger_agent = Agent(
        role="魔鬼代言人 / 蓝军（The Challenger）",
        goal="从客户领导或竞争对手视角推翻初版策略，指出时机、流程和竞争反制问题。",
        backstory=(
            "你是蓝军压力测试机。你的任务不是帮忙圆场，而是无情指出策略中的幼稚假设、"
            "流程错配、资源不足和竞争对手反击风险。你还要主动识别被忽略的内部政治、"
            "隐藏影响者和其他利益相关者，并点明销售执行时最可能缺乏的资源、能力与背书。"
        ),
        llm=shared_llm,
        verbose=debug_mode,
        allow_delegation=False,
        use_system_prompt=False,
        tools=[knowledge_tool] if knowledge_tool is not None else [],
    )

    closer_agent = Agent(
        role="销售主管 / The Closer",
        goal="从混乱讨论中先收敛简短诊断，再提炼出销售真正要执行的 3-5 条动作建议，并确保核心盲区得到验证。",
        backstory=(
            "你是只看结果的铁血销售主管。你会先用极短篇幅说清最关键的诊断结论，"
            "然后把真正该执行的动作建议按优先级排出来。"
            "如果前面有人指出核心盲区，你必须强制补上至少一条探雷动作来验证信息。"
            "你的交付重点是完整结构化收口结果，不是自由发挥。"
        ),
        llm=closer_llm or shared_llm,
        verbose=debug_mode,
        allow_delegation=False,
        use_system_prompt=False,
        tools=(
            [knowledge_tool, action_format_tool, action_executability_tool]
            if knowledge_tool is not None
            else [action_format_tool, action_executability_tool]
        ),
    )

    task_1_diagnosis = Task(
        description=prompt_specs["diagnosis"].description,
        expected_output=prompt_specs["diagnosis"].expected_output,
        agent=diagnostician_agent,
    )

    task_2_strategy = Task(
        description=prompt_specs["strategy"].description,
        expected_output=prompt_specs["strategy"].expected_output,
        agent=strategist_agent,
        context=[task_1_diagnosis],
    )

    task_3_challenge = Task(
        description=prompt_specs["challenge"].description,
        expected_output=prompt_specs["challenge"].expected_output,
        agent=challenger_agent,
        context=[task_1_diagnosis, task_2_strategy],
    )

    task_4_closer = Task(
        description=prompt_specs["closer"].description,
        expected_output=prompt_specs["closer"].expected_output,
        agent=closer_agent,
        context=[task_1_diagnosis, task_2_strategy, task_3_challenge],
    )

    return Crew(
        agents=[
            diagnostician_agent,
            strategist_agent,
            challenger_agent,
            closer_agent,
        ],
        tasks=[
            task_1_diagnosis,
            task_2_strategy,
            task_3_challenge,
            task_4_closer,
        ],
        process=Process.sequential,
        verbose=debug_mode,
    )


def run_brainstorm(
    project_name: str,
    six_element_inputs: dict[str, str],
    *,
    debug: bool = False,
    allow_prompt: bool = False,
    rules: dict[str, SixElementRule] | None = None,
) -> BrainstormResult:
    """Execute the sales battle plan from structured six-element inputs."""
    config = load_runtime_config(require_api_key=True, allow_prompt=allow_prompt)

    try:
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError as error:
        raise BrainstormExecutionError(
            "缺少 langchain_openai。请先激活虚拟环境后安装依赖：pip install langchain-openai"
        ) from error

    active_rules = rules or load_six_element_rules(config.rules_path)
    rules_context = format_rules_context(active_rules)
    project_context = format_project_context(project_name, six_element_inputs)
    knowledge_tool = build_sales_knowledge_tool(config)

    shared_llm = _build_crewai_llm(
        model=config.model_name,
        api_key=config.api_key,
        base_url=config.api_base,
        temperature=0.4,
    )
    closer_llm = _build_crewai_llm(
        model=config.model_name,
        api_key=config.api_key,
        base_url=config.api_base,
        temperature=0.0,
    )
    repair_llm = _build_chat_openai(
        ChatOpenAI,
        model=config.model_name,
        api_key=config.api_key,
        base_url=config.api_base,
        temperature=0.0,
    )

    crew = build_sales_battle_crew(
        shared_llm=shared_llm,
        debug_mode=debug or config.debug_mode,
        knowledge_tool=knowledge_tool,
        closer_llm=closer_llm,
    )
    inputs = {
        "project_name": project_name,
        "project_context": project_context,
        "rules_context": rules_context,
    }

    try:
        result = crew.kickoff(inputs=inputs)
    except Exception as error:
        raise BrainstormExecutionError(f"任务执行失败: {error}") from error

    task_outputs = list(result.tasks_output)
    diagnosis_result = task_outputs[0].raw if len(task_outputs) > 0 else ""
    strategy_result = task_outputs[1].raw if len(task_outputs) > 1 else ""
    challenge_result = task_outputs[2].raw if len(task_outputs) > 2 else ""
    closer_task_output = task_outputs[3] if len(task_outputs) > 3 else None

    closer_execution = stabilize_closer_output(
        initial_raw_text=(
            closer_task_output.raw
            if closer_task_output is not None and closer_task_output.raw
            else result.raw
        ),
        initial_structured_output=(
            closer_task_output.pydantic
            if closer_task_output is not None and closer_task_output.pydantic is not None
            else closer_task_output.json_dict if closer_task_output is not None else None
        ),
        diagnosis_result=diagnosis_result,
        strategy_result=strategy_result,
        challenge_result=challenge_result,
        repair_llm=repair_llm,
    )

    intermediate_results: dict[str, str] | None = None
    if debug:
        intermediate_results = {
            "diagnosis_result": diagnosis_result,
            "strategy_result": strategy_result,
            "challenge_result": challenge_result,
            "closer_raw_result": closer_execution.raw_result,
            "closer_rendered_result": closer_execution.rendered_result,
        }
        if closer_execution.failure_reason:
            intermediate_results["closer_failure_reason"] = closer_execution.failure_reason

    return BrainstormResult(
        project_name=project_name,
        closer_result=closer_execution.rendered_result,
        parsed_inputs=six_element_inputs,
        intermediate_results=intermediate_results,
        closer_failure_reason=closer_execution.failure_reason,
    )


def run_brainstorm_from_markdown(
    *,
    markdown_content: str,
    source_name: str = "request.md",
    project_name: str | None = None,
    debug: bool = False,
    allow_prompt: bool = False,
) -> BrainstormResult:
    """Parse markdown content and execute the brainstorm workflow."""
    resolved_project_name, six_element_inputs = parse_markdown_payload(
        markdown_content=markdown_content,
        source_name=source_name,
        project_name_override=project_name,
    )
    return run_brainstorm(
        project_name=resolved_project_name,
        six_element_inputs=six_element_inputs,
        debug=debug,
        allow_prompt=allow_prompt,
    )
