class AppError(Exception):
    """Base application exception with an HTTP status and stable error code."""

    def __init__(self, detail: str, error_code: str, status_code: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.error_code = error_code
        self.status_code = status_code


class ConfigError(AppError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail=detail, error_code="config_error", status_code=500)


class InputParseError(AppError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail=detail, error_code="input_parse_error", status_code=400)


class RulesLoadError(AppError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail=detail, error_code="rules_load_error", status_code=500)


class BrainstormExecutionError(AppError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail=detail, error_code="brainstorm_execution_error", status_code=502)
