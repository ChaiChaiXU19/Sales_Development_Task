import unittest

from langchain_core.messages import HumanMessage, SystemMessage

from app.services.crew_service import (
    _build_crewai_llm,
    _downgrade_crewai_system_messages_for_minimax,
    _downgrade_system_messages_for_minimax,
    _should_downgrade_system_messages,
)


class LLMRoleCompatibilityTests(unittest.TestCase):
    def test_minimax_provider_requires_system_message_downgrade(self) -> None:
        self.assertTrue(
            _should_downgrade_system_messages(
                "https://api.minimaxi.com/v1",
                "MiniMax-M2.7-highspeed",
            )
        )
        self.assertFalse(
            _should_downgrade_system_messages(
                "https://api.openai.com/v1",
                "gpt-4o-mini",
            )
        )

    def test_downgrade_merges_system_instruction_into_human_message(self) -> None:
        messages = [
            SystemMessage(content="你是销售主管。"),
            HumanMessage(content="请输出最终动作。"),
        ]

        downgraded = _downgrade_system_messages_for_minimax(messages)

        self.assertEqual(len(downgraded), 1)
        self.assertEqual(downgraded[0].type, "human")
        self.assertIn("系统指令", downgraded[0].content)
        self.assertIn("你是销售主管。", downgraded[0].content)
        self.assertIn("请输出最终动作。", downgraded[0].content)

    def test_downgrade_keeps_user_only_messages_unchanged(self) -> None:
        messages = [HumanMessage(content="普通请求。")]

        downgraded = _downgrade_system_messages_for_minimax(messages)

        self.assertEqual(downgraded, messages)

    def test_crewai_message_downgrade_removes_system_role(self) -> None:
        messages = [
            {"role": "system", "content": "你是销售主管。"},
            {"role": "user", "content": "请输出最终动作。"},
        ]

        downgraded = _downgrade_crewai_system_messages_for_minimax(messages)

        self.assertEqual(len(downgraded), 1)
        self.assertEqual(downgraded[0]["role"], "user")
        self.assertIn("系统指令", downgraded[0]["content"])
        self.assertIn("你是销售主管。", downgraded[0]["content"])
        self.assertIn("请输出最终动作。", downgraded[0]["content"])

    def test_minimax_crewai_llm_formats_without_system_roles(self) -> None:
        llm = _build_crewai_llm(
            model="MiniMax-M2.7-highspeed",
            api_key="test",
            base_url="https://api.minimaxi.com/v1",
            temperature=0.0,
        )

        formatted = llm._format_messages(
            [
                {"role": "system", "content": "结构化输出。"},
                {"role": "user", "content": "生成 JSON。"},
            ]
        )

        self.assertEqual([message["role"] for message in formatted], ["user"])
        self.assertIn("结构化输出。", formatted[0]["content"])
        self.assertIn("生成 JSON。", formatted[0]["content"])


if __name__ == "__main__":
    unittest.main()
