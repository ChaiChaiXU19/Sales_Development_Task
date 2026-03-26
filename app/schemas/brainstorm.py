from pydantic import BaseModel, Field


class BrainstormRequest(BaseModel):
    project_name: str | None = Field(default=None, description="可选。若为空则从 Markdown 中解析项目名称。")
    markdown_content: str = Field(..., description="包含项目名称和六要素内容的 Markdown 文本。")
    debug: bool = Field(default=False, description="是否返回中间阶段结果。")


class IntermediateResults(BaseModel):
    diagnosis_result: str | None = None
    strategy_result: str | None = None
    challenge_result: str | None = None


class BrainstormResponse(BaseModel):
    project_name: str
    closer_result: str
    parsed_inputs: dict[str, str]
    debug: bool
    intermediate_results: IntermediateResults | None = None


class HealthResponse(BaseModel):
    status: str
    rules_file_exists: bool
    model_name: str
