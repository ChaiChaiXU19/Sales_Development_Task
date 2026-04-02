from dataclasses import dataclass
from typing import Any

from app.core.config import SIX_ELEMENT_ORDER, load_runtime_config
from app.core.exceptions import BrainstormExecutionError
from app.rag.tool import build_sales_knowledge_tool
from app.services.parser import parse_markdown_payload
from app.services.rules import SixElementRule, format_rules_context, load_six_element_rules


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


OUTPUT_STYLE_GUARDRAILS = (
    "统一输出风格约束：\n"
    "1. 禁用学术体、培训体、大纲体和术语直出，不要写成内部培训材料。\n"
    "2. 禁止直接输出“直接法”“要素一”“方法论”“框架”等标签。\n"
    "3. 六要素规则库和 RAG 只能作为你的内部判断依据，要隐藏背后的理论框架，对外必须翻译成销售日常工作口语。\n"
    "4. 结论必须贴近真实推进场景，直接落到下一步动作、判断或沟通方式，不要讲理论。"
)

ACTION_STYLE_REQUIREMENTS = (
    "动作表达约束：\n"
    "1. 每条动作都必须写明 Who / When / Do What / How to say。\n"
    "2. How to say 必须是一句销售本人可以直接拿去说的话术，不能只写“去验证”“去确认”。\n"
    "3. 在给出动作前，先判断这条动作的生硬度，明确它是否像审问、施压或越级。\n"
    "4. 只要动作偏硬，就必须补一条“柔和切入点”，先给出自然由头，再进入正题。\n"
    "5. 不要把动作写成抽象建议，必须具体到怎么推进、怎么开口、怎么接下一句。"
)

CHALLENGE_STYLE_REQUIREMENTS = (
    "社交温度审查要求：\n"
    "1. 逐条指出原动作里哪些地方太硬、太直、太像审问或压人。\n"
    "2. 每条挑战意见都要给出柔和切入点、替代表达和 How to say。\n"
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
                "2. 每条动作必须包含：对应六要素、Who、When、Do What、How to say、生硬度判断、柔和切入点。\n"
                "3. 动作设计必须优先回应诊断里暴露出的高危卡点、核心盲区和核心不足，避免脱节。\n"
                "4. 只能依据六要素策略库和检索结果出招，不能凭空发挥。\n"
                "5. 动作必须可执行，避免抽象口号，所有术语都要翻译成客户沟通口语。\n"
                "6. 如果检索为空，必须明确基于当前项目上下文判断，不要伪造已有案例、客户反馈或产品能力。"
            ),
            expected_output=(
                "一份初版破局动作清单。每条动作都包含对应六要素、Who、When、Do What、How to say、"
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
                "4. 每条挑战意见必须逐条对应原动作，固定包含：原动作的问题、策略盲区揭示、执行资源不足、为什么时机不对或流程不匹配、竞争对手可能如何反制、必须修正的点、柔和切入点、替代表达、How to say。\n"
                "5. 策略盲区揭示要明确指出当前动作是否忽略了客户内部政治斗争、隐藏影响者或其他利益相关者。\n"
                "6. 执行资源不足要明确指出销售在执行该动作时最可能缺乏的资源、能力、背书或协同支持。\n"
                "7. 每条挑战意见都要明确指出原动作的生硬度问题，并补一条柔和切入点、替代表达和 How to say。\n"
                "8. 优先引用检索到的失败教训、产品能力边界和常见交付风险；若检索为空，必须明确这一点。"
            ),
            expected_output=(
                "一份口语化的蓝军挑战报告，逐条指出原动作的问题、策略盲区揭示、执行资源不足、"
                "生硬度风险、竞争反制、必须修正点，并给出柔和切入点、替代表达和 How to say。"
            ),
        ),
        "closer": TaskPromptSpec(
            description=(
                "请旁听完前面所有历史讨论后，只输出最终收口动作清单。\n\n"
                f"{OUTPUT_STYLE_GUARDRAILS}\n\n"
                f"{ACTION_STYLE_REQUIREMENTS}\n\n"
                "要求：\n"
                "0. 如提供了 sales_knowledge_search 工具，先检索与当前项目最相关的 product 与 sales 知识，再收口动作。\n"
                "1. 只提炼 3-5 条动作。\n"
                "2. 优先覆盖六要素中最薄弱、风险最高的环节。\n"
                "3. 若存在核心盲区，最终动作中必须至少包含 1 条探雷/验证信息动作，用来验证信息或消除盲区。\n"
                "4. 每条动作必须固定格式：Who: ... | When: ... | Do What: ... | How to say: ...\n"
                "5. Who 必须明确是销售本人、合作伙伴、技术人员、高层或其他角色。\n"
                "6. When 必须明确时间节点、最晚时间或阶段窗口。\n"
                "7. Do What 必须是可量化、可执行的动作，并已经吸收挑战阶段给出的柔和切入点。\n"
                "8. How to say 必须是一句可以直接复制给客户或关键人的实战话术。\n"
                "9. 优先使用检索到的成功打法、失败教训和产品能力边界来约束最终动作。\n"
                "10. 禁止输出前言、结论、分析、解释，只输出最终动作清单。"
            ),
            expected_output=(
                "仅输出 3-5 条最终动作清单，每条使用 Who / When / Do What / How to say 格式，"
                "语言口语化，不出现方法论标签，不附加任何分析说明；若存在核心盲区，动作中至少有 1 条用于验证信息。"
            ),
        ),
    }


def build_sales_battle_crew(
    shared_llm: Any,
    debug_mode: bool = False,
    knowledge_tool: Any | None = None,
):
    """Build the six-element battle crew with a closing agent."""
    try:
        from crewai import Agent, Crew, Process, Task
    except ModuleNotFoundError as error:
        raise BrainstormExecutionError(
            "缺少 crewai。请先激活虚拟环境后安装依赖：pip install crewai"
        ) from error

    prompt_specs = build_task_prompt_specs()

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
        tools=[],
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
        tools=[knowledge_tool] if knowledge_tool is not None else [],
    )

    closer_agent = Agent(
        role="销售主管 / The Closer",
        goal="从混乱讨论中提炼出销售真正要执行的 3-5 条动作，并确保核心盲区得到验证。",
        backstory=(
            "你是只看结果的铁血销售主管。你砍掉一切华丽词藻，只保留可执行动作。"
            "如果前面有人指出核心盲区，你必须强制补上至少一条探雷动作来验证信息。"
            "你的唯一交付是 Who、When、Do What 和 How to say。"
        ),
        llm=shared_llm,
        verbose=debug_mode,
        allow_delegation=False,
        tools=[knowledge_tool] if knowledge_tool is not None else [],
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
        markdown=True,
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
            "缺少 langchain_openai。请先激活虚拟环境后安装依赖：pip install langchain_openai"
        ) from error

    active_rules = rules or load_six_element_rules(config.rules_path)
    rules_context = format_rules_context(active_rules)
    project_context = format_project_context(project_name, six_element_inputs)
    knowledge_tool = build_sales_knowledge_tool(config)

    shared_llm = ChatOpenAI(
        model=config.model_name,
        api_key=config.api_key,
        base_url=config.api_base,
        temperature=0.4,
    )
    crew = build_sales_battle_crew(
        shared_llm=shared_llm,
        debug_mode=debug or config.debug_mode,
        knowledge_tool=knowledge_tool,
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

    intermediate_results: dict[str, str] | None = None
    if debug:
        task_names = [
            "diagnosis_result",
            "strategy_result",
            "challenge_result",
            "closer_result",
        ]
        intermediate_results = {
            task_name: task_output.raw
            for task_name, task_output in zip(task_names, result.tasks_output, strict=False)
        }

    return BrainstormResult(
        project_name=project_name,
        closer_result=result.raw,
        parsed_inputs=six_element_inputs,
        intermediate_results=intermediate_results,
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
