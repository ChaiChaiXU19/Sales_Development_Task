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

    def test_action_format_validator_accepts_english_format(self) -> None:
        tool = ActionFormatValidatorTool()

        result = tool._run(
            "Who: 销售经理 | When: 今天下午 4 点前 | Do What: 给客户打电话确认预算审批节点 | "
            "How to say: 我想趁今天把预算审批节奏确认一下，避免我们后面准备方向跑偏。"
        )

        self.assertEqual(result, "Format Valid: True")

    def test_action_format_validator_accepts_chinese_format(self) -> None:
        tool = ActionFormatValidatorTool()

        result = tool._run(
            "谁: 销售本人 | 时间: 本周三中午前 | 动作: 约技术负责人做 20 分钟澄清会 | "
            "话术: 我们想先把测试边界对齐，这样后面你们内部汇报会更省事。"
        )

        self.assertEqual(result, "Format Valid: True")

    def test_action_format_validator_reports_missing_talk_track(self) -> None:
        tool = ActionFormatValidatorTool()

        result = tool._run(
            "Who: 销售本人 | When: 本周内 | Do What: 约客户做预算确认 | How to say: ..."
        )

        self.assertIn("Format Valid: False", result)
        self.assertIn("How to say", result)

    def test_action_format_validator_rejects_label_only_fields(self) -> None:
        tool = ActionFormatValidatorTool()

        result = tool._run("Who:   | When: ... | Do What: 待补充 | How to say: ...")

        self.assertIn("Format Valid: False", result)
        self.assertIn("Who", result)
        self.assertIn("When", result)
        self.assertIn("Do What", result)
        self.assertIn("How to say", result)

    def test_action_executability_check_detects_one_ban_word(self) -> None:
        tool = ActionExecutabilityCheckTool()

        result = tool._run("Who: 销售本人 | When: 今天 | Do What: 加强沟通 | How to say: 我再和您同步一下。")

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
            "Who: 销售本人 | When: 明天上午 | Do What: 给采购负责人打一通 5 分钟电话确认发标时间 | "
            "How to say: 我想先把时间点对齐，方便我们内部提前准备材料。"
        )

        self.assertEqual(result, "Action Executable: True")

    def test_build_sales_battle_crew_mounts_new_tools(self) -> None:
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
