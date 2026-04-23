import unittest

from app.services.crew_service import (
    CloserActionItem,
    CloserStructuredOutput,
    _coerce_closer_structured_output,
    render_closer_result,
    stabilize_closer_output,
    validate_closer_output,
)


class FakeRepairLLM:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def with_structured_output(self, _schema, include_raw: bool = False):
        self.include_raw = include_raw
        return self

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("No fake response left for repair LLM.")
        return self.responses.pop(0)


def make_valid_output(*, probe: bool = True) -> CloserStructuredOutput:
    return CloserStructuredOutput(
        diagnosis_summary=[
            "决策链仍不清楚，拍板人与真实影响者没有核实透。",
            "流程节点临近，但竞对已提前卡住客户认知。",
        ],
        action_recommendations=[
            CloserActionItem(
                priority=3,
                action="给客户项目接口人发一条确认消息，锁定下周评审前可沟通的时间窗。",
                deadline="今天 18:00 前",
                target="客户项目接口人",
                goal="确保后续探关键人动作不会卡在时间安排上。",
                is_probe=False,
            ),
            CloserActionItem(
                priority=1,
                action="约采购接口人做 10 分钟电话，核实这轮评审后真正拍板的领导名单。",
                deadline="明天上午 11:00 前",
                target="采购接口人",
                goal="验证真实拍板链条，避免继续围着表面接口人推进。",
                is_probe=probe,
            ),
            CloserActionItem(
                priority=2,
                action="请售前和销售一起准备一页竞对差异说明，再发给技术负责人预约澄清。",
                deadline="明天下午 15:00 前",
                target="技术负责人",
                goal="提前拆掉竞对已形成的默认认知，为内部评审争取比较空间。",
                is_probe=False,
            ),
        ],
    )


class CloserStabilizationTests(unittest.TestCase):
    def test_render_closer_result_is_deterministic_and_sorted_by_priority(self) -> None:
        rendered = render_closer_result(make_valid_output())

        self.assertTrue(rendered.startswith("一、简短诊断\n- "))
        self.assertIn("\n\n二、执行动作建议\n1. 动作：约采购接口人做 10 分钟电话", rendered)
        self.assertIn("\n2. 动作：请售前和销售一起准备一页竞对差异说明", rendered)
        self.assertIn("\n3. 动作：给客户项目接口人发一条确认消息", rendered)

    def test_validate_closer_output_rejects_invalid_counts(self) -> None:
        output = CloserStructuredOutput(
            diagnosis_summary=["只有一条诊断。"],
            action_recommendations=make_valid_output().action_recommendations,
        )

        validation = validate_closer_output(output)

        self.assertFalse(validation.is_valid)
        self.assertIn("diagnosis_summary 条数必须在 2-5 条之间。", validation.errors)

    def test_validate_closer_output_rejects_incremental_trace_and_ambiguous_deadline(self) -> None:
        output = CloserStructuredOutput(
            diagnosis_summary=[
                "这里补一条诊断。",
                "流程节点还没锁定。",
            ],
            action_recommendations=[
                CloserActionItem(
                    priority=1,
                    action="补充动作五，回头再联系客户。",
                    deadline="后续推进",
                    target="客户接口人",
                    goal="继续跟进项目。",
                    is_probe=False,
                ),
                *make_valid_output().action_recommendations[1:],
            ],
        )

        validation = validate_closer_output(output)

        self.assertFalse(validation.is_valid)
        self.assertTrue(any("增量式收口痕迹" in error for error in validation.errors))
        self.assertTrue(any("模糊时间" in error for error in validation.errors))

    def test_validate_closer_output_requires_probe_action_when_blind_spots_exist(self) -> None:
        validation = validate_closer_output(
            make_valid_output(probe=False),
            diagnosis_result="核心盲区：真实拍板人仍未核实。",
            challenge_result="策略盲区揭示：忽略了隐藏影响者。",
        )

        self.assertFalse(validation.is_valid)
        self.assertIn(
            "上游识别出盲区，但最终动作中没有 is_probe=true 的探雷/验证信息动作。",
            validation.errors,
        )

    def test_stabilize_closer_output_repairs_thin_initial_result(self) -> None:
        repair_llm = FakeRepairLLM(
            responses=[
                {
                    "parsed": make_valid_output(),
                    "raw": '{"diagnosis_summary":["修复后诊断"],"action_recommendations":[...]}',
                    "parsing_error": None,
                }
            ]
        )

        result = stabilize_closer_output(
            initial_raw_text="不行.md 那种稀薄结果",
            initial_structured_output=CloserStructuredOutput(
                diagnosis_summary=["诊断太薄。", "还不够。"],
                action_recommendations=[
                    CloserActionItem(
                        priority=1,
                        action="继续跟进",
                        deadline="尽快",
                        target="客户",
                        goal="推进一下",
                        is_probe=False,
                    ),
                    CloserActionItem(
                        priority=2,
                        action="继续跟进",
                        deadline="后续",
                        target="客户",
                        goal="推进一下",
                        is_probe=False,
                    ),
                    CloserActionItem(
                        priority=3,
                        action="继续跟进",
                        deadline="后续",
                        target="客户",
                        goal="推进一下",
                        is_probe=False,
                    ),
                ],
            ),
            diagnosis_result="核心盲区：客户真实拍板人没核实。",
            strategy_result="",
            challenge_result="",
            repair_llm=repair_llm,
        )

        self.assertTrue(result.was_repaired)
        self.assertIsNone(result.failure_reason)
        self.assertEqual(len(repair_llm.prompts), 1)
        self.assertTrue(result.rendered_result.startswith("一、简短诊断\n- 决策链仍不清楚"))
        self.assertNotEqual(result.rendered_result, "不行.md 那种稀薄结果")

    def test_stabilize_closer_output_accepts_json_with_think_prefix_without_repair(self) -> None:
        repair_llm = FakeRepairLLM(responses=[])
        raw_output = (
            "<think>\n先思考一下怎么收口。\n</think>\n"
            "下面是 JSON：\n"
            f"```json\n{make_valid_output().model_dump_json()}\n```"
        )

        result = stabilize_closer_output(
            initial_raw_text=raw_output,
            initial_structured_output=None,
            diagnosis_result="核心盲区：真实拍板链仍未核实。",
            strategy_result="",
            challenge_result="",
            repair_llm=repair_llm,
        )

        self.assertFalse(result.was_repaired)
        self.assertIsNone(result.failure_reason)
        self.assertEqual(len(repair_llm.prompts), 0)
        self.assertTrue(result.rendered_result.startswith("一、简短诊断\n- 决策链仍不清楚"))

    def test_coerce_closer_output_extracts_json_from_mixed_text(self) -> None:
        raw_output = (
            "<think>模型思考内容</think>\n"
            "最终结果："
            f"{make_valid_output().model_dump_json()}"
            "\n请查收。"
        )

        parsed = _coerce_closer_structured_output(raw_output)

        self.assertIsNotNone(parsed)
        self.assertEqual(len(parsed.action_recommendations), 3)

    def test_stabilize_closer_output_falls_back_after_two_failed_repairs(self) -> None:
        invalid_output = CloserStructuredOutput(
            diagnosis_summary=["补充一条。", "还是很薄。"],
            action_recommendations=[
                CloserActionItem(
                    priority=1,
                    action="补充动作五",
                    deadline="后续",
                    target="客户",
                    goal="继续推进",
                    is_probe=False,
                ),
                CloserActionItem(
                    priority=2,
                    action="继续跟进",
                    deadline="尽快",
                    target="客户",
                    goal="继续推进",
                    is_probe=False,
                ),
                CloserActionItem(
                    priority=3,
                    action="保持联系",
                    deadline="后续",
                    target="客户",
                    goal="继续推进",
                    is_probe=False,
                ),
            ],
        )
        repair_llm = FakeRepairLLM(
            responses=[
                {
                    "parsed": invalid_output,
                    "raw": "第一次修复半成品",
                    "parsing_error": None,
                },
                {
                    "parsed": invalid_output,
                    "raw": "第二次修复半成品",
                    "parsing_error": None,
                },
            ]
        )

        result = stabilize_closer_output(
            initial_raw_text="原始 closer 半成品",
            initial_structured_output=None,
            diagnosis_result="核心盲区：仍未摸清真实拍板链。",
            strategy_result="",
            challenge_result="",
            repair_llm=repair_llm,
        )

        self.assertTrue(result.was_repaired)
        self.assertIsNotNone(result.failure_reason)
        self.assertEqual(result.rendered_result, "第二次修复半成品")
        self.assertIn("增量式收口痕迹", result.failure_reason)
        self.assertEqual(len(repair_llm.prompts), 2)


if __name__ == "__main__":
    unittest.main()
