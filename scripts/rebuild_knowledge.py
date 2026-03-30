import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import load_runtime_config
from app.rag.embeddings import EmbeddingClient
from app.rag.ingestion import rebuild_knowledge_tree
from app.rag.repository import PgVectorRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="重建 pgvector 知识库。")
    parser.add_argument(
        "--knowledge-dir",
        default=None,
        help="知识目录路径，默认使用项目根目录下的 knowledge/。",
    )
    parser.add_argument(
        "--namespace",
        action="append",
        dest="namespaces",
        help="只重建指定命名空间，可重复传入。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_runtime_config(require_api_key=False, allow_prompt=False)
    if not config.rag.is_configured:
        raise SystemExit("RAG 配置不完整，请先补齐 PostgreSQL 和 Embedding 相关环境变量。")

    knowledge_root = Path(args.knowledge_dir).expanduser().resolve() if args.knowledge_dir else config.rag.knowledge_dir
    repository = PgVectorRepository(config.rag)
    repository.ensure_schema()
    embedding_client = EmbeddingClient.from_config(config.rag)
    report = rebuild_knowledge_tree(
        knowledge_root=knowledge_root,
        repository=repository,
        embedding_client=embedding_client,
        namespaces=args.namespaces,
    )

    print(f"知识目录：{knowledge_root}")
    print(f"扫描文档：{report.scanned_documents}")
    print(f"新增文档：{report.inserted_documents}")
    print(f"更新文档：{report.updated_documents}")
    print(f"未变化文档：{report.unchanged_documents}")
    print(f"失败文档：{report.failed_documents}")
    print(f"写入切片：{report.chunk_count}")
    if report.errors:
        print("\n错误详情：")
        for error in report.errors:
            print(f"- {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
