import unittest

from app.services.crew_service import build_task_prompt_specs


class CrewServicePromptTests(unittest.TestCase):
    def test_all_task_prompts_hide_methodology_labels(self) -> None:
        prompt_specs = build_task_prompt_specs()

        for prompt_spec in prompt_specs.values():
            self.assertIn("禁用学术体", prompt_spec.description)
            self.assertIn("隐藏", prompt_spec.description)
            self.assertIn("方法论", prompt_spec.description)
            self.assertIn("直接法", prompt_spec.description)

    def test_strategy_challenge_and_closer_require_talk_track_and_soft_entry(self) -> None:
        prompt_specs = build_task_prompt_specs()

        for name in ("strategy", "challenge", "closer"):
            prompt_spec = prompt_specs[name]
            self.assertIn("How to say", prompt_spec.description)
            self.assertIn("柔和切入点", prompt_spec.description)

    def test_closer_output_format_includes_how_to_say(self) -> None:
        closer_prompt = build_task_prompt_specs()["closer"]

        self.assertIn("Who: ... | When: ... | Do What: ... | How to say: ...", closer_prompt.description)
        self.assertIn("Who / When / Do What / How to say", closer_prompt.expected_output)

    def test_diagnosis_prompt_requires_blind_spots_and_deficiencies(self) -> None:
        diagnosis_prompt = build_task_prompt_specs()["diagnosis"]

        self.assertIn("check_rule_violations", diagnosis_prompt.description)
        self.assertIn("核心盲区", diagnosis_prompt.description)
        self.assertIn("核心不足", diagnosis_prompt.description)
        self.assertIn("以为知道、但实际未经验证", diagnosis_prompt.description)
        self.assertIn("产品能力、客情关系或流程卡位", diagnosis_prompt.description)

    def test_challenge_prompt_requires_strategy_blind_spots_and_resource_gaps(self) -> None:
        challenge_prompt = build_task_prompt_specs()["challenge"]

        self.assertIn("策略盲区揭示", challenge_prompt.description)
        self.assertIn("执行资源不足", challenge_prompt.description)
        self.assertIn("客户内部政治斗争", challenge_prompt.description)
        self.assertIn("隐藏影响者", challenge_prompt.description)

    def test_closer_prompt_requires_probe_action_when_blind_spots_exist(self) -> None:
        closer_prompt = build_task_prompt_specs()["closer"]

        self.assertIn("action_format_validator", closer_prompt.description)
        self.assertIn("action_executability_check", closer_prompt.description)
        self.assertIn("未通过，必须先修正后再输出最终结果", closer_prompt.description)
        self.assertIn("若存在核心盲区", closer_prompt.description)
        self.assertIn("至少包含 1 条探雷/验证信息动作", closer_prompt.description)
        self.assertIn("至少有 1 条用于验证信息", closer_prompt.expected_output)


if __name__ == "__main__":
    unittest.main()
