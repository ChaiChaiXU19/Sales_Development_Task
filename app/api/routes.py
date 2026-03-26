from fastapi import APIRouter

from app.core.config import load_runtime_config
from app.schemas.brainstorm import BrainstormRequest, BrainstormResponse, HealthResponse, IntermediateResults
from app.services.crew_service import run_brainstorm_from_markdown


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    config = load_runtime_config(require_api_key=False, allow_prompt=False)
    rules_file_exists = config.rules_path.exists()
    status = "ok" if rules_file_exists and bool(config.api_key) else "degraded"
    return HealthResponse(
        status=status,
        rules_file_exists=rules_file_exists,
        model_name=config.model_name,
    )


@router.post("/api/v1/brainstorm", response_model=BrainstormResponse)
def brainstorm(request: BrainstormRequest) -> BrainstormResponse:
    result = run_brainstorm_from_markdown(
        markdown_content=request.markdown_content,
        source_name="request.md",
        project_name=request.project_name,
        debug=request.debug,
        allow_prompt=False,
    )

    intermediate_results = None
    if request.debug and result.intermediate_results:
        intermediate_results = IntermediateResults(
            diagnosis_result=result.intermediate_results.get("diagnosis_result"),
            strategy_result=result.intermediate_results.get("strategy_result"),
            challenge_result=result.intermediate_results.get("challenge_result"),
        )

    return BrainstormResponse(
        project_name=result.project_name,
        closer_result=result.closer_result,
        parsed_inputs=result.parsed_inputs,
        debug=request.debug,
        intermediate_results=intermediate_results,
    )
