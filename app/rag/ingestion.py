from collections.abc import Sequence
from pathlib import Path

from app.rag.chunking import build_document_chunks
from app.rag.loaders import discover_knowledge_files, load_document
from app.rag.models import EmbeddedChunk, IngestionReport


def ingest_knowledge_tree(
    *,
    knowledge_root: Path,
    repository,
    embedding_client,
    namespaces: Sequence[str] | None = None,
) -> IngestionReport:
    allowed_namespaces = set(namespaces) if namespaces else None
    report = IngestionReport()
    files = discover_knowledge_files(knowledge_root, namespaces=allowed_namespaces)
    report.scanned_documents = len(files)

    for path in files:
        try:
            document = load_document(path, knowledge_root)
            current_state = repository.get_document_state(document.namespace, document.source_path)
            if current_state is not None and current_state.sha256 == document.sha256:
                report.unchanged_documents += 1
                continue

            chunks = build_document_chunks(document)
            if not chunks:
                report.unchanged_documents += 1
                continue

            embeddings = embedding_client.embed_texts([chunk.content for chunk in chunks])
            embedded_chunks = [
                EmbeddedChunk(chunk=chunk, embedding=embedding)
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ]
            status = repository.upsert_document(
                namespace=document.namespace,
                title=document.title,
                source_path=document.source_path,
                source_type=document.source_type,
                sha256=document.sha256,
                metadata=document.metadata,
                chunks=embedded_chunks,
            )

            if status == "inserted":
                report.inserted_documents += 1
            else:
                report.updated_documents += 1
            report.chunk_count += len(embedded_chunks)
        except Exception as error:
            report.failed_documents += 1
            report.errors.append(f"{path}: {error}")

    return report


def rebuild_knowledge_tree(
    *,
    knowledge_root: Path,
    repository,
    embedding_client,
    namespaces: Sequence[str] | None = None,
) -> IngestionReport:
    if namespaces:
        repository.delete_namespaces(namespaces)
    else:
        repository.delete_all()

    return ingest_knowledge_tree(
        knowledge_root=knowledge_root,
        repository=repository,
        embedding_client=embedding_client,
        namespaces=namespaces,
    )
