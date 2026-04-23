import re
from dataclasses import dataclass
from typing import Any

from app.core.config import SIX_ELEMENT_ORDER, load_runtime_config
from app.core.exceptions import BrainstormExecutionError
from app.rag.tool import build_sales_knowledge_tool
from app.services.parser import parse_markdown_payload
from app.services.rules import SixElementRule, format_rules_context, load_six_element_rules
from app.services.tools import build_core_sales_tools


@dataclass
class BrainstormResult:
    project_name: str
    closer_result: str
    parsed_inputs: dict[str, str]
    intermediate_results: dict[str, str] | None = None


@dataclass(frozen=True)
class TaskPromptSpec:
    description: str
    expected_output: str


@dataclass(frozen=True)
class HandoffExtraction:
    tag: str
    block: str
    error: str | None = None


@dataclass
class SalesBattleCrew:
    prompt_specs: dict[str, TaskPromptSpec]
    diagnostician_agent: Any
    strategist_agent: Any
    challenger_agent: Any
    closer_agent: Any
    action_format_validator: Any
    action_executability_check: Any

    @property
    def agents(self) -> list[Any]:
        return [
            self.diagnostician_agent,
            self.strategist_agent,
            self.challenger_agent,
            self.closer_agent,
        ]


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

RISK_PLAN_STYLE_REQUIREMENTS = (
    "风险行动规划表达约束：\n"
    "1. 每个步骤必须写清 当前处境风险 / 下一步动作 / 时间期限 / 对接对象 / 核心目的。\n"
    "2. 当前处境风险必须说明未完成、未验证或未明确的关键事项，以及它导致的推进风险。\n"
    "3. 下一步动作必须是具体补救动作，不能只是“确认”“跟进”“沟通”这类单词级动作。\n"
    "4. 时间期限必须明确到最晚时间、阶段窗口或截止时点，不能只写“尽快”或“后续”。\n"
    "5. 如果动作涉及拍板、预算、采购、关系深度等敏感事项，要把柔和切入点自然写进下一步动作，"
    "但不要额外输出生硬度判断、柔和切入点字段或中间思考。"
)

CHALLENGE_STYLE_REQUIREMENTS = (
    "社交温度审查要求：\n"
    "1. 逐条指出原动作里哪些地方太硬、太直、太像审问或压人。\n"
    "2. 每条挑战意见都要给出柔和切入点和替代表达。\n"
    "3. 如果某条动作涉及敏感推进，例如拍板、预算、绩效、关系深度，优先给出顺势切入的理由。"
)

HANDOFF_TAGS = {
    "diagnosis": "STRATEGY_HANDOFF",
    "strategy": "CHALLENGE_HANDOFF",
    "challenge": "CLOSER_HANDOFF",
}

STAGE_REQUIRED_MARKERS = {
    "strategy": ["动作：", "时间期限：", "对象：", "目的："],
    "challenge": ["原动作的问题：", "必须修正的点：", "建议替代表达："],
    "closer": ["当前处境风险：", "下一步动作：", "时间期限：", "对接对象：", "核心目的："],
}

STAGE_FORBIDDEN_MARKERS = [
    "已知事实",
    "关键缺口",
    "核心盲区",
    "核心不足",
    "风险等级",
    "高危卡点",
    "综合判断",
    "诊断报告已输出完毕",
]


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
                "7. 允许用六要素做内部判断，但对外禁止显式暴露方法论框架。\n"
                "8. 诊断正文写完后，必须在末尾追加内部交接块 [[STRATEGY_HANDOFF]]...[[/STRATEGY_HANDOFF]]。\n"
                "9. 交接块只列 3-5 个最高优先级风险，不写动作、不写总结句。\n"
                "10. 交接块每个风险固定包含：六要素、高危卡点、核心盲区、核心不足、动作目标。"
            ),
            expected_output=(
                "一份口语化的诊断报告，按六要素说明已知事实、关键缺口、核心盲区、核心不足、"
                "风险等级和高危卡点，不包含任何对策、术语标签或方法论名称；"
                "报告末尾必须追加 [[STRATEGY_HANDOFF]] 内部交接块。"
            ),
        ),
        "strategy": TaskPromptSpec(
            description=(
                "请只针对诊断交接块里的高危风险，输出初版销售破局动作。\n\n"
                "【项目现状】\n{project_context}\n\n"
                "【六要素规则】\n{rules_context}\n\n"
                "【诊断交接块】\n{strategy_handoff}\n\n"
                f"{OUTPUT_STYLE_GUARDRAILS}\n\n"
                f"{ACTION_STYLE_REQUIREMENTS}\n\n"
                "角色边界：\n"
                "1. 你不是诊断师，禁止复写诊断报告。\n"
                "2. 上游 handoff 只是素材，不是你要复述的正文。\n"
                "3. 如果输出中出现 已知事实 / 关键缺口 / 核心盲区 / 核心不足 / 风险等级 / 高危卡点 / 综合判断 / 诊断报告，视为任务失败。\n"
                "4. 你只能输出动作清单和 [[CHALLENGE_HANDOFF]]...[[/CHALLENGE_HANDOFF]] 内部交接块。\n\n"
                "要求：\n"
                "0. 如提供了 sales_knowledge_search 工具，先检索与当前项目最相关的 product 与 sales 知识，再设计动作。\n"
                "1. 只针对交接块里的高危卡点出招，不要复述全部诊断。\n"
                "2. 每条动作必须包含：对应六要素、动作、时间期限、对象、目的、生硬度判断、柔和切入点。\n"
                "3. 动作设计必须优先回应交接块里的高危卡点、核心盲区和核心不足，避免脱节。\n"
                "4. 只能依据六要素策略库和检索结果出招，不能凭空发挥。\n"
                "5. 动作必须可执行，避免抽象口号，所有术语都要翻译成客户沟通口语。\n"
                "6. 如果检索为空，必须明确基于当前项目上下文判断，不要伪造已有案例、客户反馈或产品能力。\n"
                "7. 正文写完后，必须在末尾追加 [[CHALLENGE_HANDOFF]]...[[/CHALLENGE_HANDOFF]]，逐条列出动作1/动作2/... 供蓝军挑战。"
            ),
            expected_output=(
                "一份初版破局动作清单。每条动作都包含对应六要素、动作、时间期限、对象、目的、"
                "生硬度判断和柔和切入点，整体口语化且不出现诊断报告结构；"
                "正文末尾必须追加 [[CHALLENGE_HANDOFF]] 内部交接块。"
            ),
        ),
        "challenge": TaskPromptSpec(
            description=(
                "请站在客户的降本增效领导或强势竞争对手视角，对初版动作做压力测试。\n\n"
                "【项目现状】\n{project_context}\n\n"
                "【诊断交接块】\n{strategy_handoff}\n\n"
                "【初版动作交接块】\n{challenge_handoff}\n\n"
                f"{OUTPUT_STYLE_GUARDRAILS}\n\n"
                f"{CHALLENGE_STYLE_REQUIREMENTS}\n\n"
                "角色边界：\n"
                "1. 你的输入是动作，不是项目诊断任务。\n"
                "2. 你必须逐条对应 动作1 / 动作2 / ... 输出挑战意见。\n"
                "3. 禁止输出“诊断报告已输出完毕”“最薄弱要素”“最危险卡点”这类总结型诊断句。\n"
                "4. 你只能输出挑战报告和 [[CLOSER_HANDOFF]]...[[/CLOSER_HANDOFF]]，不能生成最终行动规划。\n\n"
                "你必须输出：\n"
                "1. 哪些动作太天真或时机错误\n"
                "2. 哪些动作与当前采购流程冲突\n"
                "3. 竞争对手最可能的反制方式\n"
                "4. 策略盲区揭示\n"
                "5. 执行资源不足\n"
                "6. 必须修正的点\n\n"
                "要求：\n"
                "0. 如提供了 sales_knowledge_search 工具，先检索与当前项目最相关的 product 与 sales 知识，再做压力测试。\n"
                "1. 必须逐条挑战初版动作，而不是泛泛评论。\n"
                "2. 不能生成最终方案，只能输出挑战意见和必须修正点。\n"
                "3. 挑战要结合六要素中的权重、流程阶段、竞争风险和社交温度。\n"
                "4. 每条挑战意见必须逐条对应原动作，固定包含：原动作的问题、策略盲区揭示、执行资源不足、为什么时机不对或流程不匹配、竞争对手可能如何反制、必须修正的点、柔和切入点、替代表达。\n"
                "5. 策略盲区揭示要明确指出当前动作是否忽略了客户内部政治斗争、隐藏影响者或其他利益相关者。\n"
                "6. 执行资源不足要明确指出销售在执行该动作时最可能缺乏的资源、能力、背书或协同支持。\n"
                "7. 每条挑战意见都要明确指出原动作的生硬度问题，并补一条柔和切入点和替代表达。\n"
                "8. 优先引用检索到的失败教训、产品能力边界和常见交付风险；若检索为空，必须明确这一点。\n"
                "9. 正文写完后，必须在末尾追加 [[CLOSER_HANDOFF]]...[[/CLOSER_HANDOFF]]，逐条列出修正项1/修正项2/... 给收口人使用。"
            ),
            expected_output=(
                "一份口语化的蓝军挑战报告，逐条指出原动作的问题、策略盲区揭示、执行资源不足、"
                "生硬度风险、竞争反制、必须修正点，并给出柔和切入点和替代表达；"
                "正文末尾必须追加 [[CLOSER_HANDOFF]] 内部交接块。"
            ),
        ),
        "closer": TaskPromptSpec(
            description=(
                "请基于上游提炼后的风险与修正项，输出最终《销售当前处境风险及下一步行动规划》。\n\n"
                "【项目现状】\n{project_context}\n\n"
                "【诊断交接块】\n{strategy_handoff}\n\n"
                "【蓝军修正交接块】\n{closer_handoff}\n\n"
                f"{OUTPUT_STYLE_GUARDRAILS}\n\n"
                f"{RISK_PLAN_STYLE_REQUIREMENTS}\n\n"
                "角色边界：\n"
                "1. 你的输入不是用户原始任务，而是上游提炼后的风险与修正项。\n"
                "2. 禁止输出诊断报告、摘要、综合判断、方法论说明。\n"
                "3. 如果你发现自己写成了诊断结构，必须立即改写成 3-5 个五字段步骤。\n"
                "4. 你只能输出：第X步-... / 当前处境风险 / 下一步动作 / 时间期限 / 对接对象 / 核心目的。\n\n"
                "风险行动规划判定标准：\n"
                "1. 每个步骤必须先写当前处境风险，再写下一步补救动作、时间期限、对接对象和核心目的。\n"
                "2. 当前处境风险必须写成“未完成/未验证/未明确某关键事项，导致某推进风险”的句式，不能只写问题标签或复述事实。\n"
                "3. 下一步动作必须是补救动作，不能只是“确认”“跟进”“沟通”这类单词级动作。\n"
                "4. 每个步骤必须对应一个真实缺口，优先从决策链、POC/技术结论、采购流程、竞争对手、合作伙伴协同中选择风险最高的 3-5 个。\n\n"
                "要求：\n"
                "0. 如提供了 sales_knowledge_search 工具，先检索与当前项目最相关的 product 与 sales 知识，再收口动作。\n"
                "1. 你的任务不是输出简短诊断或普通动作建议，而是输出 3-5 个风险行动规划步骤。\n"
                "2. 你必须输出完整最终版，不能假设用户已经看过前序中间结果，也不能写“补充动作五”“前文已给出”“承接上文”。\n"
                "3. 正文必须有 3-5 个步骤，并按风险优先级从高到低排列。\n"
                "4. 风险行动规划必须优先覆盖六要素中最薄弱、风险最高的环节。\n"
                "5. 若存在核心盲区，必须至少包含 1 个探雷/验证信息步骤，用来验证信息或消除盲区。\n"
                "6. 每个步骤字段固定为：当前处境风险 / 下一步动作 / 时间期限 / 对接对象 / 核心目的。\n"
                "7. 对接对象必须明确是销售本人、合作伙伴、技术人员、高层、客户关键人或其他具体角色。\n"
                "8. 时间期限必须明确，不能用“尽快”“后续”这类模糊词；下一步动作必须具体、可执行，并已经吸收挑战阶段给出的柔和切入点。\n"
                "9. 优先使用检索到的成功打法、失败教训和产品能力边界来约束最终动作。\n"
                "10. 如果信息不足，也要基于现有信息给出最稳妥的风险行动规划，不要因为信息不全而放弃输出。\n"
                "11. 只输出普通文本段落，不要输出 JSON、Markdown 表格、额外总结句或方法论标签。\n"
                "12. 诊断报告不是最终答案，最终答案必须是风险行动规划步骤。如果你发现自己只写了诊断，必须立刻改写成下面的五字段步骤。\n\n"
                "输出模板必须严格如下：\n"
                "第一步-决策链相关风险及行动规划\n"
                "当前处境风险：未明确真实拍板人、技术影响人和采购负责人，导致后续推进容易围绕表面支持者打转，存在关键决策链断点风险。\n"
                "下一步动作：预约一次15分钟信息校准沟通，借项目推进分工的由头，核实真实拍板人、技术影响人和采购负责人分别是谁。\n"
                "时间期限：4月23日下班前\n"
                "对接对象：支持者“吴”或客户项目接口人\n"
                "核心目的：摸清决策链顶端和关键影响者，避免后续动作打在错误对象上。\n\n"
                "第二步-...\n"
                "当前处境风险：...\n"
                "下一步动作：...\n"
                "时间期限：...\n"
                "对接对象：...\n"
                "核心目的：..."
            ),
            expected_output=(
                "一段普通文本格式的《销售当前处境风险及下一步行动规划》。"
                "正文为 3-5 个步骤，每个步骤固定包含 当前处境风险 / 下一步动作 / 时间期限 / 对接对象 / 核心目的；"
                "当前处境风险必须体现未完成、未验证或未明确的关键事项及其导致的推进风险。"
            ),
        ),
    }


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


def _downgrade_crewai_system_messages_for_minimax(
    messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
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


def build_sales_battle_crew(
    shared_llm: Any,
    debug_mode: bool = False,
    knowledge_tool: Any | None = None,
    closer_llm: Any | None = None,
) -> SalesBattleCrew:
    """Build the agent bundle used by the staged sales battle pipeline."""
    try:
        from crewai import Agent
    except ModuleNotFoundError as error:
        raise BrainstormExecutionError(
            "缺少 crewai。请先激活虚拟环境后安装依赖：pip install crewai"
        ) from error

    prompt_specs = build_task_prompt_specs()
    core_tools = build_core_sales_tools()
    diagnostician_tool = core_tools["check_rule_violations"]
    action_format_validator = core_tools["action_format_validator"]
    action_executability_check = core_tools["action_executability_check"]

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

    strategist_tools = [knowledge_tool] if knowledge_tool is not None else []
    strategist_agent = Agent(
        role="破局军师（The Strategist）",
        goal="针对高危卡点，严格依据六要素策略库设计初版销售破局动作。",
        backstory=(
            "你是销售破局军师，擅长把项目弱点转化为下一步推进动作。"
            "你不能空谈，必须优先回应诊断交接块暴露出的高危卡点、核心盲区和核心不足，"
            "把动作写清楚：目标、执行动作、依赖资源、触达人和预期结果。"
        ),
        llm=shared_llm,
        verbose=debug_mode,
        allow_delegation=False,
        use_system_prompt=False,
        tools=strategist_tools,
    )

    challenger_tools = [knowledge_tool] if knowledge_tool is not None else []
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
        tools=challenger_tools,
    )

    closer_tools = [action_format_validator, action_executability_check]
    if knowledge_tool is not None:
        closer_tools.insert(0, knowledge_tool)
    closer_agent = Agent(
        role="销售主管 / The Closer",
        goal="从结构化交接信息中收敛出 3-5 个销售当前处境风险及下一步行动规划步骤，并确保核心盲区得到验证。",
        backstory=(
            "你是只看结果的铁血销售主管。你不会输出普通诊断摘要，"
            "而是把当前处境风险、下一步动作、时间期限、对接对象和核心目的按普通文本步骤收口。"
            "如果前面有人指出核心盲区，你必须强制补上至少一条探雷动作来验证信息。"
            "你的交付重点是完整、可直接执行的文本收口结果，不是自由发挥。"
        ),
        llm=closer_llm or shared_llm,
        verbose=debug_mode,
        allow_delegation=False,
        use_system_prompt=False,
        tools=closer_tools,
    )

    return SalesBattleCrew(
        prompt_specs=prompt_specs,
        diagnostician_agent=diagnostician_agent,
        strategist_agent=strategist_agent,
        challenger_agent=challenger_agent,
        closer_agent=closer_agent,
        action_format_validator=action_format_validator,
        action_executability_check=action_executability_check,
    )


def _run_stage_task(
    *,
    agent: Any,
    task_prompt: TaskPromptSpec,
    inputs: dict[str, str],
    debug_mode: bool,
) -> str:
    """Run one agent as a one-task Crew so stage inputs stay explicit."""
    try:
        from crewai import Crew, Process, Task
    except ModuleNotFoundError as error:
        raise BrainstormExecutionError(
            "缺少 crewai。请先激活虚拟环境后安装依赖：pip install crewai"
        ) from error

    stage_task = Task(
        description=task_prompt.description,
        expected_output=task_prompt.expected_output,
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[stage_task],
        process=Process.sequential,
        verbose=debug_mode,
    )

    try:
        result = crew.kickoff(inputs=inputs)
    except Exception as error:
        raise BrainstormExecutionError(f"任务执行失败: {error}") from error

    task_outputs = list(getattr(result, "tasks_output", []) or [])
    stage_output = task_outputs[0].raw if task_outputs else getattr(result, "raw", "")
    if stage_output is None:
        return ""
    return str(stage_output)


def _extract_handoff_block(text: str, tag: str) -> HandoffExtraction:
    pattern = re.compile(
        rf"\[\[{re.escape(tag)}\]\](?P<body>.*?)\[\[/{re.escape(tag)}\]\]",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        error = f"Handoff Missing: 未找到 [[{tag}]]...[[/{tag}]] 交接块。"
        return HandoffExtraction(
            tag=tag,
            block=_build_missing_handoff_placeholder(tag, error),
            error=error,
        )

    return HandoffExtraction(
        tag=tag,
        block=match.group(0).strip(),
        error=None,
    )


def _build_missing_handoff_placeholder(tag: str, error: str) -> str:
    return (
        f"[[{tag}]]\n"
        f"{error}\n"
        "后续阶段只能基于有限信息保守作答，禁止复写诊断报告、诊断摘要或其他上游正文。\n"
        f"[[/{tag}]]"
    )


def stage_contract_validator(stage_name: str, text: str) -> str:
    missing_markers = [
        marker
        for marker in STAGE_REQUIRED_MARKERS.get(stage_name, [])
        if marker not in text
    ]
    forbidden_hits = [
        marker for marker in STAGE_FORBIDDEN_MARKERS if marker in text
    ] if stage_name in STAGE_REQUIRED_MARKERS else []

    handoff_tag = HANDOFF_TAGS.get(stage_name)
    missing_handoff = None
    if handoff_tag is not None and f"[[{handoff_tag}]]" not in text:
        missing_handoff = handoff_tag

    is_valid = not missing_markers and not forbidden_hits and missing_handoff is None
    lines = [f"Stage Contract Valid: {is_valid}"]
    if missing_markers:
        lines.append(f"Missing Required Markers: {missing_markers}")
    if forbidden_hits:
        lines.append(f"Forbidden Marker Hits: {forbidden_hits}")
    if missing_handoff is not None:
        lines.append(f"Missing Handoff Tag: [[{missing_handoff}]]")
    if len(lines) == 1:
        lines.append("Details: contract looks aligned with this stage.")
    return "\n".join(lines)


def _split_closer_steps(text: str) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []

    segments = re.split(r"(?=第[一二三四五六七八九十0-9]+步-)", normalized)
    steps = [segment.strip() for segment in segments if segment.strip()]
    return steps or [normalized]


def _validate_closer_steps(closer_result: str, crew_bundle: SalesBattleCrew) -> str:
    steps = _split_closer_steps(closer_result)
    if not steps:
        return "No closer steps found."

    lines: list[str] = []
    for index, step in enumerate(steps, start=1):
        format_result = crew_bundle.action_format_validator._run(step)
        executability_result = crew_bundle.action_executability_check._run(step)
        lines.append(
            "\n".join(
                [
                    f"[Step {index}]",
                    format_result,
                    executability_result,
                ]
            )
        )
    return "\n\n".join(lines)


def _run_structured_stage_pipeline(
    *,
    crew_bundle: SalesBattleCrew,
    project_name: str,
    project_context: str,
    rules_context: str,
    debug_mode: bool,
) -> tuple[str, dict[str, str]]:
    diagnosis_result = _run_stage_task(
        agent=crew_bundle.diagnostician_agent,
        task_prompt=crew_bundle.prompt_specs["diagnosis"],
        inputs={
            "project_name": project_name,
            "project_context": project_context,
            "rules_context": rules_context,
        },
        debug_mode=debug_mode,
    )
    diagnosis_validation = stage_contract_validator("diagnosis", diagnosis_result)
    strategy_handoff = _extract_handoff_block(diagnosis_result, HANDOFF_TAGS["diagnosis"])

    strategy_result = _run_stage_task(
        agent=crew_bundle.strategist_agent,
        task_prompt=crew_bundle.prompt_specs["strategy"],
        inputs={
            "project_name": project_name,
            "project_context": project_context,
            "rules_context": rules_context,
            "strategy_handoff": strategy_handoff.block,
        },
        debug_mode=debug_mode,
    )
    strategy_validation = stage_contract_validator("strategy", strategy_result)
    challenge_handoff = _extract_handoff_block(strategy_result, HANDOFF_TAGS["strategy"])

    challenge_result = _run_stage_task(
        agent=crew_bundle.challenger_agent,
        task_prompt=crew_bundle.prompt_specs["challenge"],
        inputs={
            "project_name": project_name,
            "project_context": project_context,
            "strategy_handoff": strategy_handoff.block,
            "challenge_handoff": challenge_handoff.block,
        },
        debug_mode=debug_mode,
    )
    challenge_validation = stage_contract_validator("challenge", challenge_result)
    closer_handoff = _extract_handoff_block(challenge_result, HANDOFF_TAGS["challenge"])

    closer_result = _run_stage_task(
        agent=crew_bundle.closer_agent,
        task_prompt=crew_bundle.prompt_specs["closer"],
        inputs={
            "project_name": project_name,
            "project_context": project_context,
            "strategy_handoff": strategy_handoff.block,
            "closer_handoff": closer_handoff.block,
        },
        debug_mode=debug_mode,
    )
    closer_validation = stage_contract_validator("closer", closer_result)
    closer_step_validation = _validate_closer_steps(closer_result, crew_bundle)

    intermediate_results = {
        "diagnosis_result": diagnosis_result,
        "strategy_result": strategy_result,
        "challenge_result": challenge_result,
        "closer_raw_result": closer_result,
        "strategy_handoff": strategy_handoff.block,
        "challenge_handoff": challenge_handoff.block,
        "closer_handoff": closer_handoff.block,
        "diagnosis_validation": diagnosis_validation,
        "strategy_validation": strategy_validation,
        "challenge_validation": challenge_validation,
        "closer_validation": closer_validation,
        "closer_step_validation": closer_step_validation,
    }

    for stage_name, extraction in (
        ("strategy_handoff", strategy_handoff),
        ("challenge_handoff", challenge_handoff),
        ("closer_handoff", closer_handoff),
    ):
        if extraction.error is not None:
            intermediate_results[f"{stage_name}_error"] = extraction.error

    return closer_result, intermediate_results


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

    active_rules = rules or load_six_element_rules(config.rules_path)
    rules_context = format_rules_context(active_rules)
    knowledge_tool = build_sales_knowledge_tool(config)
    project_context = format_project_context(project_name, six_element_inputs)

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

    crew = build_sales_battle_crew(
        shared_llm=shared_llm,
        debug_mode=debug or config.debug_mode,
        knowledge_tool=knowledge_tool,
        closer_llm=closer_llm,
    )

    closer_result, intermediate_results = _run_structured_stage_pipeline(
        crew_bundle=crew,
        project_name=project_name,
        project_context=project_context,
        rules_context=rules_context,
        debug_mode=debug or config.debug_mode,
    )

    return BrainstormResult(
        project_name=project_name,
        closer_result=str(closer_result),
        parsed_inputs=six_element_inputs,
        intermediate_results=intermediate_results if debug else None,
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
