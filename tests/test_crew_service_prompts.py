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

        self.assertIn("只输出普通文本段落", closer_prompt.description)
        self.assertIn("不要输出 JSON、Markdown 表格", closer_prompt.description)
        self.assertIn("第一步-决策链相关风险及行动规划", closer_prompt.description)
        self.assertIn("当前处境风险：", closer_prompt.description)
        self.assertIn("下一步动作：", closer_prompt.description)
        self.assertIn("时间期限：", closer_prompt.description)
        self.assertIn("对接对象：", closer_prompt.description)
        self.assertIn("核心目的：", closer_prompt.description)
        self.assertIn("销售当前处境风险及下一步行动规划", closer_prompt.description)
        self.assertIn("普通文本格式", closer_prompt.expected_output)

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
        self.assertIn("[[CLOSER_HANDOFF]]", challenge_prompt.description)

    def test_closer_prompt_requires_probe_action_when_blind_spots_exist(self) -> None:
        closer_prompt = build_task_prompt_specs()["closer"]

        self.assertNotIn("action_format_validator", closer_prompt.description)
        self.assertNotIn("action_executability_check", closer_prompt.description)
        self.assertIn("若存在核心盲区", closer_prompt.description)
        self.assertIn("必须至少包含 1 个探雷/验证信息步骤", closer_prompt.description)

    def test_closer_prompt_prioritizes_complete_risk_plan_restatement(self) -> None:
        closer_prompt = build_task_prompt_specs()["closer"]

        self.assertIn("不是输出简短诊断或普通动作建议", closer_prompt.description)
        self.assertIn("你必须输出完整最终版", closer_prompt.description)
        self.assertIn("不能假设用户已经看过前序中间结果", closer_prompt.description)
        self.assertIn("不能写“补充动作五”", closer_prompt.description)
        self.assertIn("正文必须有 3-5 个步骤", closer_prompt.description)
        self.assertIn("诊断报告不是最终答案", closer_prompt.description)
        self.assertIn("最终答案必须是风险行动规划步骤", closer_prompt.description)
        self.assertIn("如果信息不足，也要基于现有信息给出最稳妥的风险行动规划", closer_prompt.description)
        self.assertIn("不要输出 JSON", closer_prompt.description)

    def test_closer_prompt_requires_risk_statement_shape(self) -> None:
        closer_prompt = build_task_prompt_specs()["closer"]

        self.assertIn("当前处境风险必须写成“未完成/未验证/未明确某关键事项，导致某推进风险”", closer_prompt.description)
        self.assertIn("下一步动作必须是补救动作", closer_prompt.description)
        self.assertIn("决策链、POC/技术结论、采购流程、竞争对手、合作伙伴协同", closer_prompt.description)

    def test_stage_prompts_define_internal_handoff_contracts(self) -> None:
        prompt_specs = build_task_prompt_specs()

        self.assertIn("[[STRATEGY_HANDOFF]]", prompt_specs["diagnosis"].description)
        self.assertIn("[[CHALLENGE_HANDOFF]]", prompt_specs["strategy"].description)
        self.assertIn("[[CLOSER_HANDOFF]]", prompt_specs["challenge"].description)

    def test_strategy_prompt_explicitly_forbids_diagnosis_restatement(self) -> None:
        strategy_prompt = build_task_prompt_specs()["strategy"]

        self.assertIn("你不是诊断师", strategy_prompt.description)
        self.assertIn("上游 handoff 只是素材", strategy_prompt.description)
        self.assertIn("视为任务失败", strategy_prompt.description)
        self.assertIn("诊断报告", strategy_prompt.description)


if __name__ == "__main__":
    unittest.main()
