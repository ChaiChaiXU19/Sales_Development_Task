import re

from app.rag.models import ChunkRecord, LoadedDocument


TARGET_CHUNK_SIZE = 1000
MAX_CHUNK_SIZE = 1200
OVERLAP_SIZE = 150
SPLIT_HINTS = ("\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", "，", ",", " ")


def build_document_chunks(
    document: LoadedDocument,
    *,
    target_size: int = TARGET_CHUNK_SIZE,
    max_size: int = MAX_CHUNK_SIZE,
    overlap: int = OVERLAP_SIZE,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    chunk_index = 0

    for section in document.sections:
        section_title = str(section.metadata.get("section_title") or document.title)
        section_text = normalize_chunk_text(section.text)
        if not section_text:
            continue

        for piece in split_text_with_overlap(
            section_text,
            target_size=target_size,
            max_size=max_size,
            overlap=overlap,
        ):
            metadata = dict(document.metadata)
            metadata.update(section.metadata)
            metadata.setdefault("section_title", section_title)
            metadata.setdefault("namespace", document.namespace)
            metadata.setdefault("source_type", document.source_type)

            chunks.append(
                ChunkRecord(
                    namespace=document.namespace,
                    title=document.title,
                    source_path=document.source_path,
                    source_type=document.source_type,
                    chunk_index=chunk_index,
                    content=piece,
                    content_preview=preview_text(piece),
                    char_count=len(piece),
                    metadata=metadata,
                )
            )
            chunk_index += 1

    return chunks


def split_text_with_overlap(
    text: str,
    *,
    target_size: int = TARGET_CHUNK_SIZE,
    max_size: int = MAX_CHUNK_SIZE,
    overlap: int = OVERLAP_SIZE,
) -> list[str]:
    cleaned = normalize_chunk_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_size:
        return [cleaned]

    pieces: list[str] = []
    start = 0
    text_length = len(cleaned)
    minimum_split = max(target_size // 2, 1)

    while start < text_length:
        end = min(start + max_size, text_length)
        if end < text_length:
            split_at = find_split_point(cleaned, start, end, minimum_split)
            if split_at > start:
                end = split_at

        piece = cleaned[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= text_length:
            break

        next_start = max(end - overlap, start + 1)
        start = next_start

    return pieces


def find_split_point(text: str, start: int, end: int, minimum_split: int) -> int:
    lower_bound = min(start + minimum_split, end)
    search_window = text[lower_bound:end]
    for hint in SPLIT_HINTS:
        index = search_window.rfind(hint)
        if index != -1:
            return lower_bound + index + len(hint)
    return end


def normalize_chunk_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def preview_text(text: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."
