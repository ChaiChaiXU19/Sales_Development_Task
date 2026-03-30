import json
import tempfile
import unittest
from pathlib import Path

from app.rag.loaders import load_document


class RagLoaderTests(unittest.TestCase):
    def test_markdown_loader_builds_section_titles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            knowledge_root = Path(tmpdir)
            product_dir = knowledge_root / "product"
            product_dir.mkdir()
            path = product_dir / "guide.md"
            path.write_text("# 产品手册\n\n## 适用场景\n适合制造业客户。", encoding="utf-8")

            document = load_document(path, knowledge_root)

            self.assertEqual(document.namespace, "product")
            self.assertEqual(document.title, "产品手册")
            self.assertEqual(document.sections[0].metadata["section_title"], "产品手册 > 适用场景")

    def test_json_loader_extracts_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            knowledge_root = Path(tmpdir)
            sales_dir = knowledge_root / "sales"
            sales_dir.mkdir()
            path = sales_dir / "cases.json"
            path.write_text(
                json.dumps(
                    {
                        "title": "案例库",
                        "items": [
                            {
                                "title": "汽车客户中标复盘",
                                "summary": "提前卡位技术标准。",
                                "tags": ["制造业", "招投标"],
                                "doc_version": "2026Q1",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            document = load_document(path, knowledge_root)

            self.assertEqual(document.title, "案例库")
            self.assertEqual(document.sections[0].metadata["tags"], ["制造业", "招投标"])
            self.assertEqual(document.sections[0].metadata["doc_version"], "2026Q1")


if __name__ == "__main__":
    unittest.main()
