import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.crew_service import (
    HANDOFF_TAGS,
    _extract_handoff_block,
    _run_structured_stage_pipeline,
    build_task_prompt_specs,
    stage_contract_validator,
)
from app.services.tools import ActionExecutabilityCheckTool, ActionFormatValidatorTool


class StagePipelineTests(unittest.TestCase):
    def test_extract_handoff_block_returns_placeholder_when_missing(self) -> None:
        extraction = _extract_handoff_block("普通正文，没有交接块。", HANDOFF_TAGS["diagnosis"])

        self.assertIsNotNone(extraction.error)
        self.assertIn("Handoff Missing", extraction.error)
        self.assertIn("[[STRATEGY_HANDOFF]]", extraction.block)

    def test_stage_contract_validator_flags_diagnosis_markers_in_strategy(self) -> None:
        result = stage_contract_validator(
            "strategy",
            "已知事实：客户在测试。\n关键缺口：采购不清楚。\n[[CHALLENGE_HANDOFF]]\n动作1\n[[/CHALLENGE_HANDOFF]]",
        )

        self.assertIn("Stage Contract Valid: False", result)
        self.assertIn("Forbidden Marker Hits", result)
        self.assertIn("已知事实", result)
        self.assertIn("关键缺口", result)

    def test_structured_pipeline_only_passes_handoffs_downstream(self) -> None:
        diagnosis_output = (
            "这里是很长的诊断正文。\n"
            "已知事实：这是诊断正文，不该传给 strategy。\n"
            "[[STRATEGY_HANDOFF]]\n"
            "风险1\n六要素：决策链\n高危卡点：采购负责人未知\n核心盲区：吴能否影响采购未知\n"
            "核心不足：高层通道缺失\n动作目标：先摸清真实采购角色\n"
            "[[/STRATEGY_HANDOFF]]"
        )
        strategy_output = (
            "动作1\n对应六要素：决策链\n动作：约吴做15分钟分工校准沟通\n时间期限：今天下班前\n"
            "对象：吴\n目的：确认采购负责人和真实拍板链路\n生硬度判断：中等\n"
            "柔和切入点：借测试汇报分工顺手核实采购角色\n\n"
            "[[CHALLENGE_HANDOFF]]\n"
            "动作1\n对应风险：风险1\n对应六要素：决策链\n动作：约吴做15分钟分工校准沟通\n"
            "时间期限：今天下班前\n对象：吴\n目的：确认采购负责人和真实拍板链路\n"
            "生硬度判断：中等\n柔和切入点：借测试汇报分工顺手核实采购角色\n"
            "[[/CHALLENGE_HANDOFF]]"
        )
        challenge_output = (
            "动作1的主要问题是直接问采购负责人容易让吴有压力。\n"
            "原动作的问题：切入理由还不够自然。\n策略盲区揭示：默认吴愿意透出采购信息。\n"
            "执行资源不足：缺少高层背书。\n为什么时机不对或流程不匹配：POC结果未出，直接问采购容易过早。\n"
            "竞争对手可能如何反制：强调我方过度打听采购流程。\n必须修正的点：先用汇报分工切入，再问采购接口。\n"
            "柔和切入点：以测试汇报准备为由核对接口人。\n建议替代表达：我们想把汇报材料一次准备到位，想先确认后续流程接口是谁。\n\n"
            "[[CLOSER_HANDOFF]]\n"
            "修正项1\n对应动作：动作1\n保留/删除：保留\n原动作的问题：切入理由还不够自然\n"
            "必须修正的点：先借测试汇报准备切入，再问采购接口\n"
            "建议替代表达：我们想把汇报材料一次准备到位，想先确认后续流程接口是谁\n"
            "必须覆盖的风险：采购负责人未知\n"
            "[[/CLOSER_HANDOFF]]"
        )
        closer_output = (
            "第一步-决策链相关风险及行动规划\n"
            "当前处境风险：未明确采购负责人和真实拍板链路，导致后续推进容易围绕表面支持者打转。\n"
            "下一步动作：今天下午借测试汇报材料准备的由头，约吴做15分钟分工校准沟通，顺势确认采购接口和真实拍板链路。\n"
            "时间期限：今天下班前\n"
            "对接对象：吴\n"
            "核心目的：摸清采购角色和决策链顶端，避免后续动作打在错误对象上。"
        )

        crew_bundle = SimpleNamespace(
            prompt_specs=build_task_prompt_specs(),
            diagnostician_agent=object(),
            strategist_agent=object(),
            challenger_agent=object(),
            closer_agent=object(),
            action_format_validator=ActionFormatValidatorTool(),
            action_executability_check=ActionExecutabilityCheckTool(),
        )

        with patch(
            "app.services.crew_service._run_stage_task",
            side_effect=[diagnosis_output, strategy_output, challenge_output, closer_output],
        ) as run_stage_task:
            closer_result, intermediate_results = _run_structured_stage_pipeline(
                crew_bundle=crew_bundle,
                project_name="测试项目",
                project_context="项目正文",
                rules_context="规则正文",
                debug_mode=False,
            )

        strategy_inputs = run_stage_task.call_args_list[1].kwargs["inputs"]
        challenge_inputs = run_stage_task.call_args_list[2].kwargs["inputs"]
        closer_inputs = run_stage_task.call_args_list[3].kwargs["inputs"]

        self.assertEqual(closer_result, closer_output)
        self.assertIn("[[STRATEGY_HANDOFF]]", strategy_inputs["strategy_handoff"])
        self.assertNotIn("已知事实：这是诊断正文", strategy_inputs["strategy_handoff"])
        self.assertIn("[[CHALLENGE_HANDOFF]]", challenge_inputs["challenge_handoff"])
        self.assertNotIn("很长的诊断正文", challenge_inputs["challenge_handoff"])
        self.assertIn("[[CLOSER_HANDOFF]]", closer_inputs["closer_handoff"])
        self.assertNotIn("动作1的主要问题是直接问采购负责人容易让吴有压力", closer_inputs["closer_handoff"])
        self.assertEqual(intermediate_results["strategy_handoff_error"] if "strategy_handoff_error" in intermediate_results else "", "")
        self.assertIn("Stage Contract Valid: True", intermediate_results["closer_validation"])
        self.assertIn("Format Valid: True", intermediate_results["closer_step_validation"])


if __name__ == "__main__":
    unittest.main()
