import tempfile
import unittest
from pathlib import Path

from app.rag.ingestion import ingest_knowledge_tree
from app.rag.models import StoredDocumentState


class FakeRepository:
    def __init__(self, existing_state: StoredDocumentState | None = None) -> None:
        self.existing_state = existing_state
        self.upserts: list[dict] = []

    def get_document_state(self, namespace: str, source_path: str) -> StoredDocumentState | None:
        return self.existing_state

    def upsert_document(self, **kwargs):
        self.upserts.append(kwargs)
        return "inserted"


class FakeEmbeddingClient:
    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class RagIngestionTests(unittest.TestCase):
    def test_ingest_skips_unchanged_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            knowledge_root = Path(tmpdir)
            product_dir = knowledge_root / "product"
            product_dir.mkdir()
            file_path = product_dir / "intro.md"
            file_path.write_text("# 产品介绍\n\n这里是产品能力说明。", encoding="utf-8")

            from app.rag.loaders import compute_sha256

            repository = FakeRepository(existing_state=StoredDocumentState(document_id=1, sha256=compute_sha256(file_path)))
            report = ingest_knowledge_tree(
                knowledge_root=knowledge_root,
                repository=repository,
                embedding_client=FakeEmbeddingClient(),
            )

            self.assertEqual(report.unchanged_documents, 1)
            self.assertEqual(len(repository.upserts), 0)

    def test_ingest_embeds_and_upserts_changed_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            knowledge_root = Path(tmpdir)
            sales_dir = knowledge_root / "sales"
            sales_dir.mkdir()
            file_path = sales_dir / "playbook.txt"
            file_path.write_text("第一条经验。\n\n第二条经验。", encoding="utf-8")

            repository = FakeRepository()
            report = ingest_knowledge_tree(
                knowledge_root=knowledge_root,
                repository=repository,
                embedding_client=FakeEmbeddingClient(),
            )

            self.assertEqual(report.inserted_documents, 1)
            self.assertEqual(report.failed_documents, 0)
            self.assertEqual(len(repository.upserts), 1)


if __name__ == "__main__":
    unittest.main()
