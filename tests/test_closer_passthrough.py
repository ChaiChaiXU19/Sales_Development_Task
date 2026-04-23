import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.crew_service import run_brainstorm


class CloserPassthroughTests(unittest.TestCase):
    def test_run_brainstorm_returns_raw_closer_output_directly(self) -> None:
        config = SimpleNamespace(
            rules_path="rules.xlsx",
            model_name="fake-model",
            api_key="fake-key",
            api_base="https://example.com",
            debug_mode=False,
        )

        with (
            patch("app.services.crew_service.load_runtime_config", return_value=config),
            patch("app.services.crew_service.load_six_element_rules", return_value={}),
            patch("app.services.crew_service.format_rules_context", return_value=""),
            patch("app.services.crew_service.build_sales_knowledge_tool", return_value=None),
            patch("app.services.crew_service._build_crewai_llm", return_value=object()),
            patch("app.services.crew_service.build_sales_battle_crew", return_value=object()),
            patch(
                "app.services.crew_service._run_structured_stage_pipeline",
                return_value=(
                    "Closer 原始输出，不做校验、不 repair、不兜底。",
                    {
                        "diagnosis_result": "诊断输出",
                        "strategy_result": "策略输出",
                        "challenge_result": "挑战输出",
                        "closer_raw_result": "Closer 原始输出，不做校验、不 repair、不兜底。",
                    },
                ),
            ),
        ):
            result = run_brainstorm(
                project_name="测试项目",
                six_element_inputs={},
                debug=True,
            )

        self.assertEqual(result.closer_result, "Closer 原始输出，不做校验、不 repair、不兜底。")
        self.assertEqual(
            result.intermediate_results["closer_raw_result"],
            "Closer 原始输出，不做校验、不 repair、不兜底。",
        )


if __name__ == "__main__":
    unittest.main()
