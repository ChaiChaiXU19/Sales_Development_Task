import unittest

from app.rag.chunking import MAX_CHUNK_SIZE, OVERLAP_SIZE, build_document_chunks, split_text_with_overlap
from app.rag.models import DocumentSection, LoadedDocument


class RagChunkingTests(unittest.TestCase):
    def test_split_text_with_overlap_keeps_window_size(self) -> None:
        text = "第一段。" * 400
        chunks = split_text_with_overlap(text)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= MAX_CHUNK_SIZE for chunk in chunks))
        self.assertEqual(chunks[0][-OVERLAP_SIZE:], chunks[1][:OVERLAP_SIZE])

    def test_build_document_chunks_preserves_section_title(self) -> None:
        document = LoadedDocument(
            namespace="sales",
            title="打法库",
            source_path="sales/playbook.md",
            source_type="markdown",
            sha256="dummy",
            metadata={"doc_version": "v1"},
            sections=[DocumentSection(text="这里是一段销售打法经验。", metadata={"section_title": "冷启动"})],
        )

        chunks = build_document_chunks(document)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["section_title"], "冷启动")
        self.assertEqual(chunks[0].metadata["doc_version"], "v1")


if __name__ == "__main__":
    unittest.main()
