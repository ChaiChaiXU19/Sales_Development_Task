import getpass
import os
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import ConfigError

try:
    from dotenv import load_dotenv as _dotenv_loader
except ModuleNotFoundError:
    _dotenv_loader = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RULES_XLSX_PATH = DATA_DIR / "six_elements_rules.xlsx"
DEFAULT_API_BASE = "https://api.deepseek.com"
DEFAULT_MODEL_NAME = "deepseek-chat"
DEFAULT_CORS_ALLOW_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
DEFAULT_EMBEDDING_DIMENSION = 1024
DEFAULT_RAG_TOP_K = 5
DEFAULT_RAG_MIN_SCORE = 0.2
SIX_ELEMENT_ORDER = [
    "需求",
    "技术认可",
    "决策链",
    "竞争对手",
    "合作伙伴",
    "流程",
]


@dataclass
class RuntimeConfig:
    api_key: str
    api_base: str
    model_name: str
    debug_mode: bool
    cors_allow_origins: list[str]
    rules_path: Path
    rag: "RagConfig"


@dataclass
class RagConfig:
    pg_host: str
    pg_port: int
    pg_db: str
    pg_user: str
    pg_password: str
    embedding_api_key: str
    embedding_api_base: str
    embedding_model: str
    embedding_dimension: int
    top_k: int
    min_score: float
    knowledge_dir: Path

    @property
    def is_configured(self) -> bool:
        required_values = [
            self.pg_host,
            self.pg_db,
            self.pg_user,
            self.pg_password,
            self.embedding_api_key,
            self.embedding_model,
        ]
        return all(bool(value) for value in required_values)


def load_env_file_fallback(env_path: Path) -> None:
    """Fallback loader when python-dotenv is unavailable."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ[key] = value


def write_api_key_to_local_env(env_path: Path, api_key: str) -> None:
    """Create or update .env with a local-only API key and restrictive permissions."""
    existing_lines: list[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    updated_lines: list[str] = []
    key_written = False

    for raw_line in existing_lines:
        stripped = raw_line.strip()
        normalized = stripped[len("export ") :].strip() if stripped.startswith("export ") else stripped

        if "=" in normalized:
            current_key = normalized.split("=", 1)[0].strip()
            if current_key == "OPENAI_API_KEY":
                updated_lines.append(f"OPENAI_API_KEY={api_key}")
                key_written = True
                continue

        updated_lines.append(raw_line)

    if not key_written:
        if updated_lines and updated_lines[-1].strip():
            updated_lines.append("")
        updated_lines.append(f"OPENAI_API_KEY={api_key}")

    env_path.write_text("\n".join(updated_lines).rstrip() + "\n", encoding="utf-8")
    env_path.chmod(0o600)


def prompt_and_persist_api_key(env_path: Path) -> str:
    """Prompt once for API key, save it to .env, and restrict file permissions."""
    try:
        api_key = getpass.getpass("请输入 OPENAI/DeepSeek API Key（输入内容不会显示）: ").strip()
    except (EOFError, KeyboardInterrupt) as error:
        raise ConfigError("未获取到 API Key，已取消输入。") from error

    if not api_key:
        raise ConfigError("API Key 不能为空。")

    try:
        write_api_key_to_local_env(env_path, api_key)
    except OSError as error:
        raise ConfigError(f"无法写入本地 .env 文件：{error}") from error

    print("[INFO] API Key 已保存到本地 .env，且该文件不会进入 Git。")
    return api_key


def parse_cors_allow_origins(value: str) -> list[str]:
    origins = [item.strip() for item in value.split(",") if item.strip()]
    return origins or DEFAULT_CORS_ALLOW_ORIGINS


def parse_int_env(value: str, default: int) -> int:
    stripped = value.strip()
    if not stripped:
        return default
    try:
        return int(stripped)
    except ValueError as error:
        raise ConfigError(f"整数配置不合法：{value}") from error


def parse_float_env(value: str, default: float) -> float:
    stripped = value.strip()
    if not stripped:
        return default
    try:
        return float(stripped)
    except ValueError as error:
        raise ConfigError(f"浮点配置不合法：{value}") from error


def load_runtime_config(
    *,
    require_api_key: bool = True,
    allow_prompt: bool = False,
) -> RuntimeConfig:
    """Load runtime configuration for both API and CLI modes."""
    env_path = PROJECT_ROOT / ".env"
    if _dotenv_loader is not None:
        _dotenv_loader(dotenv_path=env_path, override=True)
    else:
        load_env_file_fallback(env_path)

    api_key = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("DEEPSEEK_API_KEY", "").strip()
        or os.getenv("API_KEY", "").strip()
        or os.getenv("api_key", "").strip()
    )
    api_base = (
        os.getenv("OPENAI_API_BASE", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
        or os.getenv("DEEPSEEK_API_BASE", "").strip()
        or DEFAULT_API_BASE
    )
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
    debug_mode = os.getenv("BRAINSTORM_DEBUG", "false").lower() == "true"
    cors_allow_origins = parse_cors_allow_origins(os.getenv("CORS_ALLOW_ORIGINS", ""))
    rag = RagConfig(
        pg_host=os.getenv("RAG_PG_HOST", "").strip(),
        pg_port=parse_int_env(os.getenv("RAG_PG_PORT", "5432"), 5432),
        pg_db=os.getenv("RAG_PG_DB", "").strip(),
        pg_user=os.getenv("RAG_PG_USER", "").strip(),
        pg_password=os.getenv("RAG_PG_PASSWORD", "").strip(),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", "").strip(),
        embedding_api_base=os.getenv("EMBEDDING_API_BASE", "").strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "").strip(),
        embedding_dimension=parse_int_env(
            os.getenv("EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION)),
            DEFAULT_EMBEDDING_DIMENSION,
        ),
        top_k=parse_int_env(os.getenv("RAG_TOP_K", str(DEFAULT_RAG_TOP_K)), DEFAULT_RAG_TOP_K),
        min_score=parse_float_env(
            os.getenv("RAG_MIN_SCORE", str(DEFAULT_RAG_MIN_SCORE)),
            DEFAULT_RAG_MIN_SCORE,
        ),
        knowledge_dir=PROJECT_ROOT / "knowledge",
    )

    if not api_key and allow_prompt and os.isatty(0):
        api_key = prompt_and_persist_api_key(env_path)

    if require_api_key and not api_key:
        raise ConfigError(
            "缺少 OPENAI_API_KEY。请在项目根目录创建 .env 并配置，或通过 CLI 首次运行时输入 key。"
        )

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = api_base
    os.environ["OPENAI_BASE_URL"] = api_base
    os.environ.setdefault("CREWAI_STORAGE_DIR", str((PROJECT_ROOT / ".runtime" / "crewai").resolve()))
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

    return RuntimeConfig(
        api_key=api_key,
        api_base=api_base,
        model_name=model_name,
        debug_mode=debug_mode,
        cors_allow_origins=cors_allow_origins,
        rules_path=RULES_XLSX_PATH,
        rag=rag,
    )
