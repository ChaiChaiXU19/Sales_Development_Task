import argparse

from app.services.crew_service import run_brainstorm_from_markdown
from app.services.parser import read_markdown_input_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于项目六要素和 CrewAI 的销售头脑风暴 CLI。",
    )
    parser.add_argument("markdown_path", help="包含项目名称和六要素内容的 Markdown 文件路径。")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="输出中间任务结果，便于调试。",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    markdown_content = read_markdown_input_file(args.markdown_path)
    result = run_brainstorm_from_markdown(
        markdown_content=markdown_content,
        source_name=args.markdown_path,
        debug=args.debug,
        allow_prompt=True,
    )

    print("=" * 80)
    print("最终《销售下一步动作计划》")
    print("=" * 80)
    print(result.closer_result)
    print("=" * 80)

    if args.debug and result.intermediate_results:
        print("\n[DEBUG] 中间任务输出如下：")
        for task_name, task_output in result.intermediate_results.items():
            print(f"\n----- {task_name} Start -----")
            print(task_output)
            print(f"----- {task_name} End -----")
        if result.closer_failure_reason:
            print("\n[DEBUG] closer_failure_reason:")
            print(result.closer_failure_reason)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
