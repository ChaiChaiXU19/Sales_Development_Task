from collections.abc import Sequence

from openai import OpenAI

from app.core.config import RagConfig


TEXT_EMBEDDING_V4_MAX_BATCH_SIZE = 10


class EmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        api_base: str,
        model: str,
        embedding_dimension: int,
        batch_size: int = TEXT_EMBEDDING_V4_MAX_BATCH_SIZE,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0。")
        self.model = model
        self.embedding_dimension = embedding_dimension
        self.batch_size = self._resolve_batch_size(model=model, batch_size=batch_size)
        client_kwargs: dict[str, str] = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base
        self.client = OpenAI(**client_kwargs)

    @classmethod
    def from_config(cls, rag_config: RagConfig) -> "EmbeddingClient":
        return cls(
            api_key=rag_config.embedding_api_key,
            api_base=rag_config.embedding_api_base,
            model=rag_config.embedding_model,
            embedding_dimension=rag_config.embedding_dimension,
        )

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

    @staticmethod
    def _resolve_batch_size(*, model: str, batch_size: int) -> int:
        if model == "text-embedding-v4":
            return min(batch_size, TEXT_EMBEDDING_V4_MAX_BATCH_SIZE)
        return batch_size

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.embedding_dimension,
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            for item in ordered:
                embedding = list(item.embedding)
                if self.embedding_dimension and len(embedding) != self.embedding_dimension:
                    raise ValueError(
                        f"Embedding 维度不匹配，期望 {self.embedding_dimension}，实际 {len(embedding)}。"
                    )
                vectors.append(embedding)
        return vectors
