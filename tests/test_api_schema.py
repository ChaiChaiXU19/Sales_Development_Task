import unittest

from app.schemas.brainstorm import BrainstormResponse


class BrainstormSchemaTests(unittest.TestCase):
    def test_brainstorm_response_keeps_only_closer_result_for_final_output(self) -> None:
        response = BrainstormResponse(
            project_name="测试项目",
            closer_result="第一步-决策链相关风险及行动规划\n当前处境风险：...",
            parsed_inputs={},
            debug=False,
        )

        dumped = response.model_dump()
        self.assertEqual(response.closer_result, "第一步-决策链相关风险及行动规划\n当前处境风险：...")
        self.assertIn("closer_result", dumped)
        self.assertNotIn("risk_plan_rows", dumped)


if __name__ == "__main__":
    unittest.main()
