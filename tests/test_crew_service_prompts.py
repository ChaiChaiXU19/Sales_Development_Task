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


if __name__ == "__main__":
    unittest.main()
