import unittest
from types import SimpleNamespace

from app.rag.embeddings import EmbeddingClient


class FakeEmbeddingsApi:
    def __init__(self, response_vectors=None, *, dimension: int = 3):
        self.response_vectors = response_vectors
        self.dimension = dimension
        self.calls = []
        self._embedding_seed = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        vectors = self.response_vectors
        if vectors is None:
            batch = kwargs["input"]
            vectors = []
            for _ in batch:
                self._embedding_seed += 1
                vectors.append([float(self._embedding_seed)] * self.dimension)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=vector)
                for index, vector in enumerate(vectors)
            ]
        )


class RagEmbeddingTests(unittest.TestCase):
    def test_embed_texts_passes_dimensions_to_api(self) -> None:
        client = EmbeddingClient(
            api_key="dummy",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="text-embedding-v4",
            embedding_dimension=1024,
        )
        fake_api = FakeEmbeddingsApi([[0.1] * 1024])
        client.client = SimpleNamespace(embeddings=fake_api)

        vectors = client.embed_texts(["你好，世界"])

        self.assertEqual(len(vectors), 1)
        self.assertEqual(fake_api.calls[0]["model"], "text-embedding-v4")
        self.assertEqual(fake_api.calls[0]["dimensions"], 1024)

    def test_embed_texts_raises_when_dimension_mismatches(self) -> None:
        client = EmbeddingClient(
            api_key="dummy",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="text-embedding-v4",
            embedding_dimension=1024,
        )
        fake_api = FakeEmbeddingsApi([[0.1] * 512])
        client.client = SimpleNamespace(embeddings=fake_api)

        with self.assertRaisesRegex(ValueError, "Embedding 维度不匹配"):
            client.embed_texts(["维度校验"])

    def test_text_embedding_v4_batches_requests_by_ten(self) -> None:
        client = EmbeddingClient(
            api_key="dummy",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="text-embedding-v4",
            embedding_dimension=3,
            batch_size=32,
        )
        fake_api = FakeEmbeddingsApi(dimension=3)
        client.client = SimpleNamespace(embeddings=fake_api)

        vectors = client.embed_texts([f"文本 {index}" for index in range(11)])

        self.assertEqual(client.batch_size, 10)
        self.assertEqual(len(vectors), 11)
        self.assertEqual(len(fake_api.calls), 2)
        self.assertEqual(len(fake_api.calls[0]["input"]), 10)
        self.assertEqual(len(fake_api.calls[1]["input"]), 1)


if __name__ == "__main__":
    unittest.main()
