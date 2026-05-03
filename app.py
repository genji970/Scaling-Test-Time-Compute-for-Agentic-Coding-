
import os
import subprocess
from pathlib import Path

import gradio as gr


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_demo():
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    model_name = os.getenv("MSWEA_MODEL_NAME", "gemini/gemini-3-pro-preview")

    if not gemini_api_key:
        return (
            "GEMINI_API_KEY is not set.\n\n"
            "Hugging Face Space에서 Settings → Variables and secrets → New secret에 "
            "GEMINI_API_KEY를 추가해야 합니다."
        )

    cmd = [
        "python",
        "experiment/run.py",
        "--provider", "gemini",
        "--model_name", model_name,
        "--gemini_api_key", gemini_api_key,
        "--dataset_name", "princeton-nlp/SWE-bench_Verified",
        "--train_split", "test",
        "--max_dataset_samples", "1",
        "--start_sample_index", "0",
        "--end_sample_index", "1",
        "--num_rollouts", "2",
        "--num_iterations", "2",
        "--top_k", "1",
        "--group_size", "2",
        "--vote_count", "1",
        "--max_steps_per_rollout", "10",
        "--agent_max_steps", "1",
        "--pdr_max_summary_chars", "4000",
        "--pdr_max_context_chars", "16000",
        "--mini_swe_agent_extra_args=--yolo",
        "--swebench_prepare_workspace",
        "--swebench_enabled",
        "--swebench_dry_run",
        "--swebench_max_workers", "1",
        "--swebench_cache_level", "env",
        "--continue_on_error",
        "--experiment_output_path", "outputs/pdr_rtv_results.jsonl",
        "--experiment_summary_path", "outputs/pdr_rtv_summary.json",
    ]

    safe_cmd = [
        "********" if x == gemini_api_key else x
        for x in cmd
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=900,
        )

        output = ""
        output += "## Command\n\n"
        output += "```bash\n" + " ".join(safe_cmd) + "\n```\n\n"

        output += "## Return code\n\n"
        output += f"```text\n{result.returncode}\n```\n\n"

        output += "## stdout\n\n"
        output += "```text\n" + result.stdout[-12000:] + "\n```\n\n"

        if result.stderr:
            output += "## stderr\n\n"
            output += "```text\n" + result.stderr[-8000:] + "\n```\n\n"

        summary_path = OUTPUT_DIR / "pdr_rtv_summary.json"
        if summary_path.exists():
            output += "## Summary JSON\n\n"
            output += "```json\n"
            output += summary_path.read_text(encoding="utf-8")[-8000:]
            output += "\n```\n"

        return output

    except subprocess.TimeoutExpired:
        return (
            "Demo timed out.\n\n"
            "Space에서는 전체 benchmark가 아니라 작은 dry-run만 돌리는 게 좋습니다."
        )


with gr.Blocks(title="Scaling Test-Time Compute for Agentic Coding Demo") as demo:
    gr.Markdown(
        """
# Scaling Test-Time Compute for Agentic Coding — Unofficial Demo

This Space runs a tiny demo of the PDR + RTV pipeline.

It uses:

- SWE-Bench Verified
- 1 sample
- 2 rollouts
- 2 iterations
- dry-run mode

For full experiments, clone the GitHub repository and run locally or on RunPod.
"""
    )

    run_button = gr.Button("Run demo")
    output = gr.Markdown()

    run_button.click(
        fn=run_demo,
        inputs=[],
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch()
