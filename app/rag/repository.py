from collections.abc import Sequence
from contextlib import contextmanager

from app.core.config import RagConfig
from app.rag.models import EmbeddedChunk, SearchResult, StoredDocumentState


class PgVectorRepository:
    def __init__(self, rag_config: RagConfig) -> None:
        self.rag_config = rag_config

    @contextmanager
    def connect(self):
        try:
            import psycopg
        except ModuleNotFoundError as error:
            raise RuntimeError("缺少 psycopg，请先安装 PostgreSQL 驱动依赖。") from error

        connection = psycopg.connect(
            host=self.rag_config.pg_host,
            port=self.rag_config.pg_port,
            dbname=self.rag_config.pg_db,
            user=self.rag_config.pg_user,
            password=self.rag_config.pg_password,
        )
        try:
            yield connection
        finally:
            connection.close()

    def ping(self) -> None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()

    def ensure_schema(self) -> None:
        dimension = self.rag_config.embedding_dimension
        if dimension <= 0:
            raise ValueError("EMBEDDING_DIMENSION 必须大于 0。")

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS knowledge_documents (
                        id BIGSERIAL PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        title TEXT NOT NULL,
                        source_path TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        sha256 TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE(namespace, source_path)
                    )
                    """
                )
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS knowledge_chunks (
                        id BIGSERIAL PRIMARY KEY,
                        document_id BIGINT NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        content_preview TEXT NOT NULL,
                        embedding vector({dimension}) NOT NULL,
                        char_count INTEGER NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE(document_id, chunk_index)
                    )
                    """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_knowledge_documents_sha256 ON knowledge_documents (namespace, sha256)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_knowledge_documents_namespace ON knowledge_documents (namespace)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_document_id ON knowledge_chunks (document_id)"
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding_ivfflat
                    ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                    """
                )
            connection.commit()

    def get_document_state(self, namespace: str, source_path: str) -> StoredDocumentState | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, sha256
                    FROM knowledge_documents
                    WHERE namespace = %s AND source_path = %s
                    """,
                    (namespace, source_path),
                )
                row = cursor.fetchone()
        if row is None:
            return None
        return StoredDocumentState(document_id=row[0], sha256=row[1])

    def upsert_document(
        self,
        *,
        namespace: str,
        title: str,
        source_path: str,
        source_type: str,
        sha256: str,
        metadata: dict,
        chunks: Sequence[EmbeddedChunk],
    ) -> str:
        try:
            from psycopg.types.json import Jsonb
        except ModuleNotFoundError as error:
            raise RuntimeError("缺少 psycopg.types.json，请先安装 psycopg。") from error

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, sha256
                    FROM knowledge_documents
                    WHERE namespace = %s AND source_path = %s
                    """,
                    (namespace, source_path),
                )
                existing = cursor.fetchone()

                if existing is None:
                    cursor.execute(
                        """
                        INSERT INTO knowledge_documents (
                            namespace,
                            title,
                            source_path,
                            source_type,
                            sha256,
                            metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (namespace, title, source_path, source_type, sha256, Jsonb(metadata)),
                    )
                    document_id = cursor.fetchone()[0]
                    status = "inserted"
                else:
                    document_id = existing[0]
                    status = "updated"
                    cursor.execute(
                        """
                        UPDATE knowledge_documents
                        SET title = %s,
                            source_type = %s,
                            sha256 = %s,
                            metadata = %s,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (title, source_type, sha256, Jsonb(metadata), document_id),
                    )
                    cursor.execute("DELETE FROM knowledge_chunks WHERE document_id = %s", (document_id,))

                if chunks:
                    cursor.executemany(
                        """
                        INSERT INTO knowledge_chunks (
                            document_id,
                            chunk_index,
                            content,
                            content_preview,
                            embedding,
                            char_count,
                            metadata
                        )
                        VALUES (%s, %s, %s, %s, %s::vector, %s, %s)
                        """,
                        [
                            (
                                document_id,
                                item.chunk.chunk_index,
                                item.chunk.content,
                                item.chunk.content_preview,
                                format_vector(item.embedding),
                                item.chunk.char_count,
                                Jsonb(item.chunk.metadata),
                            )
                            for item in chunks
                        ],
                    )
            connection.commit()
        return status

    def delete_all(self) -> None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("TRUNCATE knowledge_documents CASCADE")
            connection.commit()

    def delete_namespaces(self, namespaces: Sequence[str]) -> None:
        if not namespaces:
            return
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM knowledge_documents WHERE namespace = ANY(%s)",
                    (list(namespaces),),
                )
            connection.commit()

    def search(
        self,
        *,
        query_embedding: list[float],
        namespaces: Sequence[str] | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        query = """
            WITH query_embedding AS (SELECT %s::vector AS embedding)
            SELECT
                d.title,
                d.namespace,
                d.source_path,
                d.source_type,
                c.content_preview,
                c.metadata,
                1 - (c.embedding <=> q.embedding) AS score
            FROM knowledge_chunks c
            JOIN knowledge_documents d ON d.id = c.document_id
            CROSS JOIN query_embedding q
        """

        filters = ["1 - (c.embedding <=> q.embedding) >= %s"]
        params: list[object] = [format_vector(query_embedding), min_score]

        if namespaces:
            filters.append("d.namespace = ANY(%s)")
            params.append(list(namespaces))

        query += f" WHERE {' AND '.join(filters)}"
        query += " ORDER BY c.embedding <=> q.embedding LIMIT %s"
        params.append(top_k)

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()

        return [
            SearchResult(
                title=row[0],
                namespace=row[1],
                source_path=row[2],
                source_type=row[3],
                content_preview=row[4],
                metadata=row[5] or {},
                score=float(row[6]),
            )
            for row in rows
        ]


def format_vector(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.10f}" for value in values) + "]"
