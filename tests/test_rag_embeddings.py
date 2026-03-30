import unittest
from types import SimpleNamespace

from app.rag.embeddings import EmbeddingClient


class FakeEmbeddingsApi:
    def __init__(self, response_vectors):
        self.response_vectors = response_vectors
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=vector)
                for index, vector in enumerate(self.response_vectors)
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


if __name__ == "__main__":
    unittest.main()
