import os
from pathlib import Path
from typing import Type

from pydantic import BaseModel, Field, PrivateAttr

from app.core.config import RuntimeConfig
from app.rag.service import KnowledgeSearchService, format_results_for_llm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = (PROJECT_ROOT / ".runtime").resolve()
RUNTIME_HOME = (RUNTIME_DIR / "home").resolve()
RUNTIME_HOME.mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(RUNTIME_HOME)
os.environ.setdefault("XDG_DATA_HOME", str((RUNTIME_DIR / "xdg-data").resolve()))
os.environ.setdefault("XDG_CONFIG_HOME", str((RUNTIME_DIR / "xdg-config").resolve()))
if "CREWAI_STORAGE_DIR" not in os.environ:
    os.environ["CREWAI_STORAGE_DIR"] = str((RUNTIME_DIR / "crewai").resolve())
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai.tools import BaseTool


class SalesKnowledgeSearchInput(BaseModel):
    query: str = Field(..., description="要检索的产品知识、案例经验或销售打法问题。")
    namespace: str | None = Field(
        default=None,
        description="可选。限定为 product 或 sales；为空时同时检索两个知识域。",
    )
    top_k: int = Field(default=5, ge=1, le=10, description="最多返回多少条候选片段。")


class SalesKnowledgeSearchTool(BaseTool):
    name: str = "sales_knowledge_search"
    description: str = (
        "检索产品知识和真实销售经验。"
        "在设计策略、做蓝军挑战、收口最终动作前先使用此工具，"
        "优先获取相关的产品能力边界、成功打法、失败教训和交付约束。"
    )
    args_schema: Type[BaseModel] = SalesKnowledgeSearchInput
    _service: KnowledgeSearchService = PrivateAttr()

    def __init__(self, *, service: KnowledgeSearchService, **data):
        super().__init__(**data)
        self._service = service

    def _run(self, query: str, namespace: str | None = None, top_k: int = 5) -> str:
        namespaces = [namespace] if namespace else ["product", "sales"]
        results = self._service.search(query, namespaces=namespaces, top_k=top_k)
        return format_results_for_llm(results)


def build_sales_knowledge_tool(config: RuntimeConfig) -> SalesKnowledgeSearchTool | None:
    if not config.rag.is_configured:
        return None

    service = KnowledgeSearchService.from_runtime_config(config)
    try:
        service.ping()
    except Exception:
        return None
    return SalesKnowledgeSearchTool(service=service)
