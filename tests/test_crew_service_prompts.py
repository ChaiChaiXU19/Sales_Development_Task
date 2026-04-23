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

    def test_strategy_challenge_and_closer_require_new_action_fields_and_soft_entry(self) -> None:
        prompt_specs = build_task_prompt_specs()

        for name in ("strategy", "challenge", "closer"):
            prompt_spec = prompt_specs[name]
            self.assertIn("柔和切入点", prompt_spec.description)
        for name in ("strategy", "closer"):
            prompt_spec = prompt_specs[name]
            self.assertIn("时间期限", prompt_spec.description)
            self.assertIn("对象", prompt_spec.description)
            self.assertIn("目的", prompt_spec.description)

    def test_closer_output_format_uses_new_contract(self) -> None:
        closer_prompt = build_task_prompt_specs()["closer"]

        self.assertIn("只输出一个 JSON 对象", closer_prompt.description)
        self.assertIn("diagnosis_summary", closer_prompt.description)
        self.assertIn("action_recommendations", closer_prompt.description)
        self.assertIn("priority / action / deadline / target / goal / is_probe", closer_prompt.description)
        self.assertIn("完整 JSON 对象", closer_prompt.expected_output)

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
        self.assertIn("至少包含 1 条 is_probe=true 的探雷/验证信息动作", closer_prompt.description)
        self.assertIn("至少 1 条动作的 is_probe 为 true", closer_prompt.expected_output)

    def test_closer_prompt_prioritizes_complete_structured_restatement(self) -> None:
        closer_prompt = build_task_prompt_specs()["closer"]

        self.assertIn("不是只做风险诊断，而是先给出简短诊断结论", closer_prompt.description)
        self.assertIn("你必须输出完整最终版", closer_prompt.description)
        self.assertIn("不能假设用户已经看过 strategy/challenge", closer_prompt.description)
        self.assertIn("不能写“补充动作五”", closer_prompt.description)
        self.assertIn("action_recommendations 必须有 3-5 条", closer_prompt.description)
        self.assertIn("如果信息不足，也要基于现有信息给出最稳妥的下一步动作", closer_prompt.description)
        self.assertIn("不要输出 Markdown", closer_prompt.description)
        self.assertIn("JSON 顶层字段只能是 diagnosis_summary 和 action_recommendations", closer_prompt.description)


if __name__ == "__main__":
    unittest.main()
