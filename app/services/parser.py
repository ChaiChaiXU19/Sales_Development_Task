import re
from pathlib import Path

from app.core.config import SIX_ELEMENT_ORDER
from app.core.exceptions import InputParseError


def read_markdown_input_file(file_path: str) -> str:
    """Read a single markdown file that contains the whole project context."""
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if not path.exists():
        raise InputParseError(f"文件不存在：{path}")
    if not path.is_file():
        raise InputParseError(f"不是普通文件：{path}")
    if path.suffix.lower() != ".md":
        raise InputParseError("仅支持读取 .md 文件。")

    try:
        content = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as error:
        raise InputParseError("Markdown 文件不是有效的 UTF-8 文本，请转换编码后重试。") from error

    if not content:
        raise InputParseError("Markdown 文件内容为空，请重新提供。")

    return content


def parse_markdown_project_input(markdown_text: str, source_name: str) -> tuple[str, dict[str, str]]:
    """Parse one markdown document into project name and six-element inputs."""
    normalized = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    sections: dict[str, list[str]] = {name: [] for name in SIX_ELEMENT_ORDER}
    current_section: str | None = None
    project_name = ""

    project_name_patterns = [
        re.compile(r"^\s*项目名称\s*[:：]\s*(.+?)\s*$"),
        re.compile(r"^\s*#\s*项目名称\s*[:：]?\s*(.+?)\s*$"),
        re.compile(r"^\s*##\s*项目名称\s*[:：]?\s*(.+?)\s*$"),
    ]

    heading_pattern = re.compile(r"^\s{0,3}(#{1,6})\s*(.+?)\s*$")

    for line in normalized.split("\n"):
        stripped = line.strip()

        if not project_name:
            for pattern in project_name_patterns:
                match = pattern.match(stripped)
                if match:
                    project_name = match.group(1).strip()
                    break

        heading_match = heading_pattern.match(line)
        if heading_match:
            heading_text = heading_match.group(2).strip()
            matched_section = next(
                (name for name in SIX_ELEMENT_ORDER if heading_text == name),
                None,
            )
            if matched_section is not None:
                current_section = matched_section
                continue

        if current_section is not None:
            sections[current_section].append(line)

    if not project_name:
        first_heading = next(
            (
                match.group(2).strip()
                for raw_line in normalized.split("\n")
                if (match := heading_pattern.match(raw_line)) and match.group(1) == "#"
            ),
            "",
        )
        project_name = first_heading or Path(source_name).stem

    parsed_inputs = {
        name: "\n".join(lines).strip() or "暂无输入（待补充）"
        for name, lines in sections.items()
    }

    missing_sections = [
        name for name, content in parsed_inputs.items() if content == "暂无输入（待补充）"
    ]
    if len(missing_sections) == len(SIX_ELEMENT_ORDER):
        raise InputParseError(
            "Markdown 文件未识别到六要素内容。请使用二级标题，例如：## 需求、## 技术认可。"
        )

    return project_name, parsed_inputs


def parse_markdown_payload(
    markdown_content: str,
    *,
    source_name: str = "request.md",
    project_name_override: str | None = None,
) -> tuple[str, dict[str, str]]:
    """Parse request markdown and apply an optional project name override."""
    if not markdown_content.strip():
        raise InputParseError("markdown_content 不能为空。")

    parsed_project_name, six_element_inputs = parse_markdown_project_input(
        markdown_text=markdown_content,
        source_name=source_name,
    )
    project_name = (project_name_override or "").strip() or parsed_project_name
    return project_name, six_element_inputs
