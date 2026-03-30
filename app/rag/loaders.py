import hashlib
import json
import re
from pathlib import Path
from typing import Any

from app.rag.models import DocumentSection, LoadedDocument


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".json"}
JSON_TEXT_FIELDS = [
    "title",
    "summary",
    "content",
    "body",
    "problem",
    "solution",
    "lesson",
    "lessons",
    "playbook",
    "notes",
]
JSON_META_FIELDS = {"tags", "doc_version", "author", "owner", "category"}
HEADING_PATTERN = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")


def normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def compute_sha256(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def discover_knowledge_files(knowledge_root: Path, namespaces: set[str] | None = None) -> list[Path]:
    if not knowledge_root.exists():
        return []

    allowed_namespaces = namespaces or None
    files: list[Path] = []
    for path in sorted(knowledge_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        relative = path.relative_to(knowledge_root)
        if len(relative.parts) < 2:
            continue

        namespace = relative.parts[0]
        if allowed_namespaces and namespace not in allowed_namespaces:
            continue

        files.append(path)
    return files


def load_document(path: Path, knowledge_root: Path) -> LoadedDocument:
    relative = path.relative_to(knowledge_root)
    namespace = relative.parts[0]
    suffix = path.suffix.lower()

    if suffix == ".md":
        return load_markdown_document(path, knowledge_root, namespace)
    if suffix == ".txt":
        return load_text_document(path, knowledge_root, namespace)
    if suffix == ".pdf":
        return load_pdf_document(path, knowledge_root, namespace)
    if suffix == ".json":
        return load_json_document(path, knowledge_root, namespace)

    raise ValueError(f"不支持的知识文件类型：{path}")


def load_markdown_document(path: Path, knowledge_root: Path, namespace: str) -> LoadedDocument:
    raw_text = path.read_text(encoding="utf-8")
    normalized = normalize_text(raw_text)
    lines = normalized.splitlines()
    title = path.stem
    sections: list[DocumentSection] = []
    heading_stack: dict[int, str] = {}
    current_heading = "概览"
    current_lines: list[str] = []

    for line in lines:
        match = HEADING_PATTERN.match(line)
        if match:
            flushed_section = flush_section(current_heading, current_lines)
            if flushed_section is not None:
                sections.append(flushed_section)

            level = len(match.group(1))
            heading_text = match.group(2).strip()
            if level == 1 and title == path.stem:
                title = heading_text

            heading_stack = {key: value for key, value in heading_stack.items() if key < level}
            heading_stack[level] = heading_text
            current_heading = " > ".join(heading_stack[index] for index in sorted(heading_stack))
            current_lines = []
            continue

        current_lines.append(line)

    flushed_section = flush_section(current_heading, current_lines)
    if flushed_section is not None:
        sections.append(flushed_section)

    return LoadedDocument(
        namespace=namespace,
        title=title,
        source_path=str(path.relative_to(knowledge_root)),
        source_type="markdown",
        sha256=compute_sha256(path),
        metadata={"namespace": namespace, "source_type": "markdown"},
        sections=sections or [DocumentSection(text=normalized, metadata={"section_title": title})],
    )


def load_text_document(path: Path, knowledge_root: Path, namespace: str) -> LoadedDocument:
    normalized = normalize_text(path.read_text(encoding="utf-8"))
    sections = paragraph_sections(normalized, fallback_title=path.stem)
    return LoadedDocument(
        namespace=namespace,
        title=path.stem,
        source_path=str(path.relative_to(knowledge_root)),
        source_type="text",
        sha256=compute_sha256(path),
        metadata={"namespace": namespace, "source_type": "text"},
        sections=sections,
    )


def load_pdf_document(path: Path, knowledge_root: Path, namespace: str) -> LoadedDocument:
    try:
        import pdfplumber
    except ModuleNotFoundError as error:
        raise RuntimeError("缺少 pdfplumber，无法加载 PDF 知识文件。") from error

    page_texts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            page_text = normalize_text(page.extract_text() or "")
            if not page_text:
                continue
            page_texts.append(page_text)

    normalized = normalize_text("\n\n".join(page_texts))
    sections = paragraph_sections(normalized, fallback_title=path.stem)
    return LoadedDocument(
        namespace=namespace,
        title=path.stem,
        source_path=str(path.relative_to(knowledge_root)),
        source_type="pdf",
        sha256=compute_sha256(path),
        metadata={"namespace": namespace, "source_type": "pdf", "page_count": len(page_texts)},
        sections=sections,
    )


def load_json_document(path: Path, knowledge_root: Path, namespace: str) -> LoadedDocument:
    data = json.loads(path.read_text(encoding="utf-8"))
    title = path.stem
    metadata: dict[str, Any] = {"namespace": namespace, "source_type": "json"}
    sections: list[DocumentSection] = []

    if isinstance(data, dict):
        title = str(data.get("title") or data.get("name") or title)
        for key in JSON_META_FIELDS:
            if key in data and data[key] not in (None, "", []):
                metadata[key] = data[key]

        list_payload = None
        for candidate in ("items", "records", "cases", "documents"):
            value = data.get(candidate)
            if isinstance(value, list):
                list_payload = value
                break

        if list_payload is not None:
            sections = build_json_sections(list_payload, title)
        else:
            sections = build_json_sections([data], title)
    elif isinstance(data, list):
        sections = build_json_sections(data, title)
    else:
        sections = [DocumentSection(text=normalize_text(json.dumps(data, ensure_ascii=False)), metadata={})]

    return LoadedDocument(
        namespace=namespace,
        title=title,
        source_path=str(path.relative_to(knowledge_root)),
        source_type="json",
        sha256=compute_sha256(path),
        metadata=metadata,
        sections=sections,
    )


def build_json_sections(items: list[Any], fallback_title: str) -> list[DocumentSection]:
    sections: list[DocumentSection] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            lines: list[str] = []
            section_metadata: dict[str, Any] = {}
            section_title = str(
                item.get("title")
                or item.get("name")
                or item.get("summary")
                or f"{fallback_title}-{index}"
            )

            for field_name in JSON_TEXT_FIELDS:
                value = item.get(field_name)
                if value in (None, "", []):
                    continue
                rendered = render_json_value(value)
                if rendered:
                    lines.append(f"{field_name}: {rendered}" if field_name != "content" else rendered)

            for key in JSON_META_FIELDS:
                value = item.get(key)
                if value not in (None, "", []):
                    section_metadata[key] = value

            rendered_text = normalize_text("\n".join(lines) or json.dumps(item, ensure_ascii=False))
            if rendered_text:
                section_metadata["section_title"] = section_title
                sections.append(DocumentSection(text=rendered_text, metadata=section_metadata))
            continue

        rendered_text = normalize_text(render_json_value(item))
        if rendered_text:
            sections.append(
                DocumentSection(
                    text=rendered_text,
                    metadata={"section_title": f"{fallback_title}-{index}"},
                )
            )

    return sections or [DocumentSection(text=fallback_title, metadata={"section_title": fallback_title})]


def render_json_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "；".join(str(item) for item in value if item not in (None, ""))
    return json.dumps(value, ensure_ascii=False)


def paragraph_sections(text: str, fallback_title: str) -> list[DocumentSection]:
    paragraphs = [item.strip() for item in re.split(r"\n{2,}", text) if item.strip()]
    if not paragraphs:
        return [DocumentSection(text=fallback_title, metadata={"section_title": fallback_title})]
    return [
        DocumentSection(text=paragraph, metadata={"section_title": fallback_title})
        for paragraph in paragraphs
    ]


def flush_section(section_title: str, lines: list[str]) -> DocumentSection | None:
    section_text = normalize_text("\n".join(lines))
    if not section_text:
        return None
    return DocumentSection(text=section_text, metadata={"section_title": section_title})
