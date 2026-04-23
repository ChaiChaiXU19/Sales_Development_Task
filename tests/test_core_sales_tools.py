import unittest

from crewai import LLM

from app.services.crew_service import build_sales_battle_crew
from app.services.tools import (
    ActionExecutabilityCheckTool,
    ActionFormatValidatorTool,
    CheckRuleViolationsTool,
)


class CoreSalesToolsTests(unittest.TestCase):
    def test_check_rule_violations_detects_missing_decision_maker(self) -> None:
        tool = CheckRuleViolationsTool()

        result = tool._run("当前项目决策人不清楚，而且还没见领导。")

        self.assertIn("决策链", result)
        self.assertIn("决策链缺失关键KP", result)
        self.assertIn("高层关系未建立", result)

    def test_check_rule_violations_detects_competitor_lock_in(self) -> None:
        tool = CheckRuleViolationsTool()

        result = tool._run("客户已经内定竞争对手，对手关系很深。")

        self.assertIn("竞争对手", result)
        self.assertIn("竞争对手已被锁定", result)

    def test_check_rule_violations_returns_no_issue_for_neutral_text(self) -> None:
        tool = CheckRuleViolationsTool()

        result = tool._run("客户已经安排下周技术交流，预算范围和关键人都比较清晰。")

        self.assertEqual(result, "已识别的违规项：无明显违规")

    def test_check_rule_violations_handles_empty_text(self) -> None:
        tool = CheckRuleViolationsTool()

        result = tool._run("   ")

        self.assertIn("Check Rule Violations Error", result)

    def test_action_format_validator_accepts_new_format(self) -> None:
        tool = ActionFormatValidatorTool()

        result = tool._run(
            "动作: 给客户打电话确认预算审批节点 | 时间期限: 今天下午 4 点前 | "
            "对象: 采购负责人 | 目的: 确认预算审批节奏，避免后续准备方向跑偏。"
        )

        self.assertEqual(result, "Format Valid: True")

    def test_action_format_validator_accepts_common_deadline_alias(self) -> None:
        tool = ActionFormatValidatorTool()

        result = tool._run(
            "动作: 约技术负责人做 20 分钟澄清会 | 截止时间: 本周三中午前 | "
            "对象: 技术负责人 | 目的: 对齐测试边界，方便客户内部汇报。"
        )

        self.assertEqual(result, "Format Valid: True")

    def test_action_format_validator_reports_missing_purpose(self) -> None:
        tool = ActionFormatValidatorTool()

        result = tool._run(
            "动作: 约客户做预算确认 | 时间期限: 本周内 | 对象: 客户采购接口人 | 目的: ..."
        )

        self.assertIn("Format Valid: False", result)
        self.assertIn("目的", result)

    def test_action_format_validator_rejects_label_only_fields(self) -> None:
        tool = ActionFormatValidatorTool()

        result = tool._run("动作: 待补充 | 时间期限: ... | 对象:   | 目的: ...")

        self.assertIn("Format Valid: False", result)
        self.assertIn("动作", result)
        self.assertIn("时间期限", result)
        self.assertIn("对象", result)
        self.assertIn("目的", result)

    def test_action_executability_check_detects_one_ban_word(self) -> None:
        tool = ActionExecutabilityCheckTool()

        result = tool._run("动作: 加强沟通 | 时间期限: 今天 | 对象: 客户项目接口人 | 目的: 推进关系。")

        self.assertIn("Executability Warning", result)
        self.assertIn("加强沟通", result)

    def test_action_executability_check_detects_multiple_ban_words(self) -> None:
        tool = ActionExecutabilityCheckTool()

        result = tool._run("先持续跟进，再保持联系，后面逐步建立信任。")

        self.assertIn("持续跟进", result)
        self.assertIn("保持联系", result)
        self.assertIn("建立信任", result)

    def test_action_executability_check_accepts_specific_action(self) -> None:
        tool = ActionExecutabilityCheckTool()

        result = tool._run(
            "动作: 给采购负责人打一通 5 分钟电话确认发标时间 | 时间期限: 明天上午 | "
            "对象: 采购负责人 | 目的: 对齐发标时间，便于内部提前准备材料。"
        )

        self.assertEqual(result, "Action Executable: True")

    def test_action_format_validator_accepts_closer_step_contract(self) -> None:
        tool = ActionFormatValidatorTool()

        result = tool._run(
            "第一步-决策链相关风险及行动规划\n"
            "当前处境风险：未明确采购负责人，导致后续推进存在决策链断点风险。\n"
            "下一步动作：今天下午约支持者做 15 分钟信息校准沟通，确认采购负责人是谁。\n"
            "时间期限：今天下班前\n"
            "对接对象：支持者吴\n"
            "核心目的：摸清采购角色，避免后续动作打在错误对象上。"
        )

        self.assertEqual(result, "Format Valid: True")

    def test_action_executability_check_detects_quick_follow_up_phrase(self) -> None:
        tool = ActionExecutabilityCheckTool()

        result = tool._run(
            "下一步动作：尽快跟进采购负责人，确认发标时间。\n"
            "时间期限：明天中午前\n"
            "对接对象：采购负责人\n"
            "核心目的：确认流程时间表。"
        )

        self.assertIn("Executability Warning", result)
        self.assertIn("尽快跟进", result)

    def test_build_sales_battle_crew_mounts_closer_validation_tools(self) -> None:
        llm = LLM(model="openai/gpt-4o-mini", api_key="test", base_url="https://example.com/v1")

        crew = build_sales_battle_crew(shared_llm=llm, knowledge_tool=None)

        diagnostician_tools = [tool.name for tool in crew.agents[0].tools]
        closer_tools = [tool.name for tool in crew.agents[3].tools]

        self.assertEqual(diagnostician_tools, ["check_rule_violations"])
        self.assertEqual(
            closer_tools,
            ["action_format_validator", "action_executability_check"],
        )


if __name__ == "__main__":
    unittest.main()
