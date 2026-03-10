import os
from pathlib import Path
from typing import Any

try:
    # 优先使用 python-dotenv（如果已安装）。
    from dotenv import load_dotenv as _dotenv_loader
except ModuleNotFoundError:
    _dotenv_loader = None


def load_env_file_fallback(env_path: Path) -> None:
    """当 python-dotenv 不可用时，使用简易解析器加载 .env。"""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ[key] = value


def load_runtime_config() -> dict[str, str]:
    """加载 .env 中的配置，并转换为 CrewAI/OpenAI 兼容的环境变量。"""
    env_path = Path(__file__).resolve().parent / ".env"
    if _dotenv_loader is not None:
        # override=True: 防止系统环境里存在空值导致 .env 无法生效
        _dotenv_loader(dotenv_path=env_path, override=True)
    else:
        load_env_file_fallback(env_path)

    # 兼容多种 key 命名，优先读取标准 OPENAI_API_KEY。
    api_key = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("DEEPSEEK_API_KEY", "").strip()
        or os.getenv("API_KEY", "").strip()
        or os.getenv("api_key", "").strip()
    )
    api_base = (
        os.getenv("OPENAI_API_BASE", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
        or os.getenv("DEEPSEEK_API_BASE", "").strip()
    )
    model_name = os.getenv("MODEL_NAME", "deepseek-chat").strip()

    # 最后兜底默认 DeepSeek 接口地址，减少环境配置负担。
    if not api_base:
        api_base = "https://api.deepseek.com"

    # 如果仍缺 key，允许在终端手动输入，避免直接报错退出。
    if not api_key and os.isatty(0):
        api_key = input("请输入 OPENAI/DeepSeek API Key: ").strip()

    if not api_key:
        raise ValueError(
            "缺少 OPENAI_API_KEY。请在项目根目录创建 .env 并配置，或在运行时输入 key。\n"
            "示例：OPENAI_API_KEY=你的key"
        )

    # CrewAI 的 OpenAI provider 默认读取 OPENAI_BASE_URL，这里做一次映射。
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = api_base
    os.environ["OPENAI_BASE_URL"] = api_base
    os.environ.setdefault(
        "CREWAI_STORAGE_DIR",
        str((Path(__file__).resolve().parent / ".runtime" / "crewai").resolve()),
    )
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

    return {
        "api_key": api_key,
        "api_base": api_base,
        "model_name": model_name,
    }


def build_sales_brainstorm_crew(shared_llm: Any):
    """构建 3 个 Agent + 3 个 Task 的顺序执行 Crew。"""
    # 延迟导入，确保先完成 .env 加载与环境变量映射。
    try:
        from crewai import Agent, Crew, Process, Task
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "缺少 crewai。请先激活虚拟环境后安装依赖：\n"
            "source .venv/bin/activate && pip install crewai langchain_openai python-dotenv"
        ) from error

    creative_agent = Agent(
        role="首席创意销售策略师",
        goal="针对给定产品和目标客户，提出 3 个大胆且具吸引力的销售切入点。",
        backstory=(
            "你是富有远见的销售天才，擅长打破常规吸引客户注意。"
            "你不考虑预算和实施限制，只追求最大胆的创意。"
        ),
        llm=shared_llm,
        verbose=True,
        allow_delegation=False,
    )

    realist_agent = Agent(
        role="资深销售运营与风控总监",
        goal="对创意方案逐一做现实剖析，识别风险与落地障碍。",
        backstory=(
            "你是务实甚至悲观的销售老将，重视 ROI、销售周期与实施成本。"
            "你的职责是识别华而不实方案中的致命问题。"
        ),
        llm=shared_llm,
        verbose=True,
        allow_delegation=False,
    )

    facilitator_agent = Agent(
        role="销售战役总指挥（VP of Sales）",
        goal="综合创意与批评，产出可直接执行的销售行动指南。",
        backstory=(
            "你能平衡创新与执行，把混乱讨论整理为一线销售可直接照做的 SOP。"
        ),
        llm=shared_llm,
        verbose=True,
        allow_delegation=False,
    )

    task_1_creative = Task(
        description=(
            "你需要基于以下输入进行创意发散：\n"
            "- 产品名称: {product_name}\n"
            "- 目标客户画像: {target_client}\n"
            "- 客户核心痛点: {client_pain_point}\n\n"
            "请输出 3 个天马行空、极具吸引力的销售切入点和营销噱头。"
            "每个方案都要有：方案名称、核心创意、销售切入话术。"
        ),
        expected_output=(
            "输出 3 个编号清晰的创意方案（1/2/3），每个方案包含名称、思路与销售话术。"
        ),
        agent=creative_agent,
    )

    task_2_realist = Task(
        description=(
            "请对上一步的 3 个创意方案进行逐一现实评估。\n"
            "每个方案必须包含：\n"
            "1) 成本风险\n"
            "2) 落地难度\n"
            "3) 客户可能的反驳点\n"
            "4) 技术可行性判断\n"
            "请保持直接、犀利、可执行。"
        ),
        expected_output=(
            "按方案 1/2/3 输出风险评估报告，每个方案至少给出 4 个维度的结论。"
        ),
        agent=realist_agent,
        context=[task_1_creative],
    )

    task_3_facilitator = Task(
        description=(
            "请综合创意方案与风险评估，输出最终《销售实战行动指南》Markdown 报告。\n"
            "报告必须包含以下 3 个一级标题：\n"
            "# 核心销售主张\n"
            "# 预案Q&A（客户反驳与应对）\n"
            "# 行动建议（下一步该干什么）\n\n"
            "要求：保留创意亮点、规避关键风险、给出可执行 SOP。"
        ),
        expected_output=(
            "一份结构化 Markdown 报告，能被销售团队直接拿去执行。"
        ),
        agent=facilitator_agent,
        context=[task_1_creative, task_2_realist],
        markdown=True,
    )

    return Crew(
        agents=[creative_agent, realist_agent, facilitator_agent],
        tasks=[task_1_creative, task_2_realist, task_3_facilitator],
        process=Process.sequential,
        verbose=True,
    )


def run_brainstorm(product_name: str, target_client: str, client_pain_point: str) -> None:
    """执行一次完整的多智能体销售头脑风暴。"""
    config = load_runtime_config()

    try:
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "缺少 langchain_openai。请先激活虚拟环境后安装依赖：\n"
            "source .venv/bin/activate && pip install crewai langchain_openai python-dotenv"
        ) from error

    print("[INFO] 已加载 .env 配置")
    print(f"[INFO] 使用模型: {config['model_name']}")
    print(f"[INFO] 使用 API Base: {config['api_base']}")

    # 使用 langchain_openai 创建共享模型实例，再交给 CrewAI Agent 使用。
    shared_llm = ChatOpenAI(
        model=config["model_name"],
        api_key=config["api_key"],
        base_url=config["api_base"],
        temperature=0.7,
    )

    print("[INFO] 正在创建 3 个销售智能体与任务链...")
    crew = build_sales_brainstorm_crew(shared_llm=shared_llm)

    inputs = {
        "product_name": product_name,
        "target_client": target_client,
        "client_pain_point": client_pain_point,
    }

    print("[INFO] 开始顺序执行任务（Creative -> Realist -> Facilitator）...")
    print("[INFO] 输入参数如下：")
    for key, value in inputs.items():
        print(f"  - {key}: {value}")

    try:
        result = crew.kickoff(inputs=inputs)
    except Exception as error:
        print(f"[ERROR] 任务执行失败: {error}")
        return

    print("\n[INFO] 全部任务执行完成，任务摘要如下：")
    for i, task_output in enumerate(result.tasks_output, start=1):
        print(f"\n----- Task {i} Output Start -----")
        print(task_output.raw)
        print(f"----- Task {i} Output End -----")

    print("\n" + "=" * 80)
    print("最终《销售实战行动指南》（Markdown）")
    print("=" * 80)
    print(result.raw)
    print("=" * 80)


if __name__ == "__main__":
    def ask_required_input(prompt_text: str) -> str:
        """询问并校验必填输入。"""
        while True:
            value = input(prompt_text).strip()
            if value:
                return value
            print("[WARN] 输入不能为空，请重新输入。")

    print("[INFO] 请输入本次头脑风暴的业务信息：")
    try:
        product_name = ask_required_input("产品名称：")
        target_client = ask_required_input("目标客户画像：")
        client_pain_point = ask_required_input("客户核心痛点：")
    except (KeyboardInterrupt, EOFError):
        print("\n[INFO] 用户取消输入，程序结束。")
        raise SystemExit(0)

    run_brainstorm(
        product_name=product_name,
        target_client=target_client,
        client_pain_point=client_pain_point,
    )
