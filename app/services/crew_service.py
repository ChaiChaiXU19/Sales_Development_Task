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


def format_project_context(project_name: str, six_element_inputs: dict[str, str]) -> str:
    """Render project inputs into a stable prompt block."""
    blocks = [f"项目名称：{project_name}"]
    for name in SIX_ELEMENT_ORDER:
        blocks.append(f"【{name}】\n{six_element_inputs.get(name, '暂无输入（待补充）')}")
    return "\n\n".join(blocks)


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

    diagnostician_agent = Agent(
        role="情报侦察兵（The Diagnostician）",
        goal="严格对照项目六要素标准，找出当前项目情报缺口、风险和高危卡点。",
        backstory=(
            "你是专门挑刺的销售情报侦察兵。你不负责想对策，只负责对照标准找盲区、"
            "识别高危卡点，并按风险高低排序。"
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
            "你不能空谈，必须把动作写清楚：目标、执行动作、依赖资源、触达人和预期结果。"
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
            "流程错配、资源不足和竞争对手反击风险。"
        ),
        llm=shared_llm,
        verbose=debug_mode,
        allow_delegation=False,
        tools=[knowledge_tool] if knowledge_tool is not None else [],
    )

    closer_agent = Agent(
        role="销售主管 / The Closer",
        goal="从混乱讨论中提炼出销售真正要执行的 3-5 条动作。",
        backstory=(
            "你是只看结果的铁血销售主管。你砍掉一切华丽词藻，只保留可执行动作。"
            "你的唯一交付是 Who、When、Do What。"
        ),
        llm=shared_llm,
        verbose=debug_mode,
        allow_delegation=False,
        tools=[knowledge_tool] if knowledge_tool is not None else [],
    )

    task_1_diagnosis = Task(
        description=(
            "请基于以下项目现状和六要素规则，做一份高质量诊断报告。\n\n"
            "【项目现状】\n{project_context}\n\n"
            "【六要素规则】\n{rules_context}\n\n"
            "要求：\n"
            "1. 必须按六要素逐项输出。\n"
            "2. 每个要素固定包含：已知事实、关键缺口、风险等级、高危卡点。\n"
            "3. 只做诊断，不允许提出策略、建议、应对措施。\n"
            "4. 结合权重，明确当前最薄弱的要素和最危险的卡点。"
        ),
        expected_output=(
            "一份按六要素结构化输出的诊断报告，只包含事实、缺口、风险和卡点，不包含任何对策。"
        ),
        agent=diagnostician_agent,
    )

    task_2_strategy = Task(
        description=(
            "请只针对上一步识别出的高危卡点，输出初版销售破局动作。\n\n"
            "你必须同时参考以下规则库：\n{rules_context}\n\n"
            "要求：\n"
            "0. 如提供了 sales_knowledge_search 工具，先检索与当前项目最相关的 product 与 sales 知识，再设计动作。\n"
            "1. 只针对高危卡点出招，不要复述全部诊断。\n"
            "2. 每条动作必须包含：对应六要素、动作目标、执行动作、依赖资源、触达人、预期结果。\n"
            "3. 只能依据六要素策略库出招，不能凭空发挥。\n"
            "4. 动作必须可执行，避免抽象口号。\n"
            "5. 如果检索为空，必须明确基于当前项目上下文判断，不要伪造已有案例、客户反馈或产品能力。"
        ),
        expected_output=(
            "一份初版破局动作清单，每条动作都包含对应要素、目标、动作、依赖资源、触达人和预期结果。"
        ),
        agent=strategist_agent,
        context=[task_1_diagnosis],
    )

    task_3_challenge = Task(
        description=(
            "请站在客户的降本增效领导或强势竞争对手视角，对上一步初版策略做压力测试。\n\n"
            "你必须输出：\n"
            "1. 哪些动作太天真或时机错误\n"
            "2. 哪些动作与当前采购流程冲突\n"
            "3. 竞争对手最可能的反制方式\n"
            "4. 必须修正的点\n\n"
            "要求：\n"
            "0. 如提供了 sales_knowledge_search 工具，先检索与当前项目最相关的 product 与 sales 知识，再做压力测试。\n"
            "1. 必须逐条挑战初版动作，而不是泛泛评论。\n"
            "2. 不能生成最终方案，只能输出挑战意见和必须修正点。\n"
            "3. 挑战要结合六要素中的权重、流程阶段和竞争风险。\n"
            "4. 优先引用检索到的失败教训、产品能力边界和常见交付风险；若检索为空，必须明确这一点。"
        ),
        expected_output=(
            "一份蓝军挑战报告，包含对初版动作的逐条反驳、风险说明和必须修正点。"
        ),
        agent=challenger_agent,
        context=[task_1_diagnosis, task_2_strategy],
    )

    task_4_closer = Task(
        description=(
            "请旁听完前面所有历史讨论后，只输出最终收口动作清单。\n\n"
            "要求：\n"
            "0. 如提供了 sales_knowledge_search 工具，先检索与当前项目最相关的 product 与 sales 知识，再收口动作。\n"
            "1. 只提炼 3-5 条动作。\n"
            "2. 优先覆盖六要素中最薄弱、风险最高的环节。\n"
            "3. 每条动作必须固定格式：Who: ... | When: ... | Do What: ...\n"
            "4. Who 必须明确是销售本人、合作伙伴、技术人员、高层或其他角色。\n"
            "5. When 必须明确时间节点、最晚时间或阶段窗口。\n"
            "6. Do What 必须是可量化、可执行的动作，不允许空话。\n"
            "7. 优先使用检索到的成功打法、失败教训和产品能力边界来约束最终动作。\n"
            "8. 禁止输出前言、结论、分析、解释，只输出最终动作清单。"
        ),
        expected_output=(
            "仅输出 3-5 条最终动作清单，每条使用 Who / When / Do What 格式，不包含其他说明。"
        ),
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
