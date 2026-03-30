from collections.abc import Sequence

from openai import OpenAI

from app.core.config import RagConfig


class EmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        api_base: str,
        model: str,
        embedding_dimension: int,
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.embedding_dimension = embedding_dimension
        self.batch_size = batch_size
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
