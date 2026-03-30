from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DocumentSection:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedDocument:
    namespace: str
    title: str
    source_path: str
    source_type: str
    sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: list[DocumentSection] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return Path(self.source_path)


@dataclass
class ChunkRecord:
    namespace: str
    title: str
    source_path: str
    source_type: str
    chunk_index: int
    content: str
    content_preview: str
    char_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddedChunk:
    chunk: ChunkRecord
    embedding: list[float]


@dataclass
class StoredDocumentState:
    document_id: int
    sha256: str


@dataclass
class SearchResult:
    title: str
    namespace: str
    source_path: str
    source_type: str
    score: float
    content_preview: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionReport:
    scanned_documents: int = 0
    inserted_documents: int = 0
    updated_documents: int = 0
    unchanged_documents: int = 0
    failed_documents: int = 0
    chunk_count: int = 0
    errors: list[str] = field(default_factory=list)
