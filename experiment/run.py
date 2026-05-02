import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = PROJECT_ROOT / "main"

if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from args import args
from pipeline import PDRRTVPipeline


logging.basicConfig(level=logging.INFO)
logging.info("PDR+RTV experiment initialized.")


class PDRRTVExperiment:
    """
    여러 dataset sample에 대해 PDR+RTV pipeline을 반복 실행하는 실험 파일.

    main/pipeline.py:
        sample 하나에 대해 PDR+RTV 실행

    experiment/run_pdr_rtv.py:
        여러 sample에 대해 pipeline.run_sample(sample)을 반복 실행
        결과를 jsonl로 저장
    """

    def __init__(self):
        self.pipeline = PDRRTVPipeline()
        self.results: List[Dict[str, Any]] = []

    def select_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        args.start_sample_index, args.end_sample_index 기준으로 실행할 sample 범위를 자른다.
        """

        start = max(args.start_sample_index, 0)

        if args.end_sample_index is None or args.end_sample_index < 0:
            end = len(samples)
        else:
            end = min(args.end_sample_index, len(samples))

        if start >= end:
            raise ValueError(
                f"Invalid sample range: start={start}, end={end}, total={len(samples)}"
            )

        return samples[start:end]

    def save_jsonl_record(self, record: Dict[str, Any], output_path: str) -> None:
        """
        sample 하나의 결과를 jsonl 파일에 append한다.
        """

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_summary(self, summary: Dict[str, Any], output_path: str) -> None:
        """
        전체 실험 summary를 json으로 저장한다.
        """

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def build_experiment_summary(self) -> Dict[str, Any]:
        """
        현재는 evaluator가 없으므로 실행 성공/실패와 rollout 상태 중심으로 summary를 만든다.
        """

        total = len(self.results)
        completed = sum(1 for result in self.results if result.get("error") is None)
        failed = total - completed

        best_status_counts: Dict[str, int] = {}

        for result in self.results:
            status = result.get("best_status", "error")
            best_status_counts[status] = best_status_counts.get(status, 0) + 1

        return {
            "total_samples": total,
            "completed_samples": completed,
            "failed_samples": failed,
            "best_status_counts": best_status_counts,
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "split": args.train_split,
            "num_rollouts": args.num_rollouts,
            "top_k": args.top_k,
            "group_size": args.group_size,
            "vote_count": args.vote_count,
            "num_iterations": args.num_iterations,
            "agent_max_steps": args.agent_max_steps,
            "model_name": args.model_name,
            "endpoint_url": args.endpoint_url,
        }

    def run(self) -> Dict[str, Any]:
        """
        전체 experiment 실행.
        """

        samples = self.pipeline.load_samples()
        selected_samples = self.select_samples(samples)

        output_path = Path(args.experiment_output_path)
        if output_path.exists():
            output_path.unlink()

        logging.info(f"Running experiment on {len(selected_samples)} samples.")

        for local_idx, sample in enumerate(selected_samples):
            global_idx = sample.get("index", local_idx)

            logging.info(
                f"Running sample {local_idx + 1}/{len(selected_samples)} "
                f"(global index={global_idx})"
            )

            try:
                result = self.pipeline.run_sample(sample)
                result["error"] = None

            except Exception as error:
                logging.exception(f"Sample failed: {global_idx}")

                if not args.continue_on_error:
                    raise

                result = {
                    "sample_index": global_idx,
                    "instance_id": sample.get("instance_id"),
                    "problem_statement": sample.get("problem_statement"),
                    "error": f"{type(error).__name__}: {error}",
                }

            self.results.append(result)
            self.save_jsonl_record(result, args.experiment_output_path)

        summary = self.build_experiment_summary()
        self.save_summary(summary, args.experiment_summary_path)

        logging.info(f"Saved results to: {args.experiment_output_path}")
        logging.info(f"Saved summary to: {args.experiment_summary_path}")

        return summary


if __name__ == "__main__":
    experiment = PDRRTVExperiment()
    summary = experiment.run()

    print("Experiment finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))