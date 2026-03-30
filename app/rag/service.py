from collections.abc import Sequence

from app.core.config import RagConfig, RuntimeConfig
from app.rag.embeddings import EmbeddingClient
from app.rag.models import SearchResult
from app.rag.repository import PgVectorRepository


class KnowledgeSearchService:
    def __init__(
        self,
        *,
        repository: PgVectorRepository,
        embedding_client: EmbeddingClient,
        default_top_k: int,
        default_min_score: float,
    ) -> None:
        self.repository = repository
        self.embedding_client = embedding_client
        self.default_top_k = default_top_k
        self.default_min_score = default_min_score

    @classmethod
    def from_runtime_config(cls, config: RuntimeConfig) -> "KnowledgeSearchService":
        return cls.from_rag_config(config.rag)

    @classmethod
    def from_rag_config(cls, rag_config: RagConfig) -> "KnowledgeSearchService":
        return cls(
            repository=PgVectorRepository(rag_config),
            embedding_client=EmbeddingClient.from_config(rag_config),
            default_top_k=rag_config.top_k,
            default_min_score=rag_config.min_score,
        )

    def ping(self) -> None:
        self.repository.ping()

    def search(
        self,
        query: str,
        *,
        namespaces: Sequence[str] | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        query_embedding = self.embedding_client.embed_query(query)
        return self.repository.search(
            query_embedding=query_embedding,
            namespaces=namespaces,
            top_k=top_k or self.default_top_k,
            min_score=self.default_min_score if min_score is None else min_score,
        )


def format_results_for_llm(results: Sequence[SearchResult]) -> str:
    if not results:
        return (
            "未检索到相关产品知识或销售经验。请仅基于当前项目上下文和六要素规则继续分析，"
            "并明确不要虚构已有案例或产品能力。"
        )

    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"{index}. [{result.source_type}/{result.namespace}] {result.title}",
                    f"相关片段：{result.content_preview}",
                    f"相似度：{result.score:.3f}",
                    f"来源路径：{result.source_path}",
                ]
            )
        )
    return "\n\n".join(blocks)
