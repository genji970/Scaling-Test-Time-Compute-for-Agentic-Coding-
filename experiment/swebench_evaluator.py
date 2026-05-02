import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from main.args import args


logging.basicConfig(level=logging.INFO)
logging.info("SWE-bench evaluator initialized.")


@dataclass
class SWEBenchPrediction:
    instance_id: str
    model_name_or_path: str
    model_patch: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "instance_id": self.instance_id,
            "model_name_or_path": self.model_name_or_path,
            "model_patch": self.model_patch,
        }


class SWEBenchEvaluator:
    """
    SWE-bench official harness wrapper.

    역할:
    1. repo workspace에서 git diff를 추출한다.
    2. SWE-bench prediction jsonl 형식으로 저장한다.
    3. swebench.harness.run_evaluation을 실행한다.
    4. evaluation_results 안의 결과 파일을 찾아 읽는다.

    주의:
    - 이 evaluator는 repo checkout/setup을 직접 하지 않는다.
    - agent/executor가 작업한 repo_dir이 이미 SWE-bench instance의 workspace여야 한다.
    - 진짜 SWE-bench 평가를 하려면 Docker와 swebench package가 설치되어 있어야 한다.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        predictions_path: str,
        run_id: str,
        max_workers: int,
        cache_level: str,
        clean: bool,
        results_dir: str,
        modal: bool = False,
        dry_run: bool = False,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.predictions_path = Path(predictions_path)
        self.run_id = run_id
        self.max_workers = max_workers
        self.cache_level = cache_level
        self.clean = clean
        self.results_dir = Path(results_dir)
        self.modal = modal
        self.dry_run = dry_run

    def extract_patch(self, repo_dir: str) -> str:
        """
        agent가 수정한 repo workspace에서 git diff를 추출한다.
        """

        repo_path = Path(repo_dir).resolve()

        if not repo_path.exists():
            raise FileNotFoundError(f"Repository directory does not exist: {repo_path}")

        command = ["git", "diff", "--binary"]

        logging.info(f"Extracting git diff from: {repo_path}")

        completed = subprocess.run(
            command,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "Failed to extract git diff.\n"
                f"STDOUT:\n{completed.stdout}\n\n"
                f"STDERR:\n{completed.stderr}"
            )

        patch = completed.stdout

        if not patch.strip():
            logging.warning("Extracted patch is empty.")

        return patch

    def build_prediction(
        self,
        instance_id: str,
        model_patch: str,
        model_name_or_path: Optional[str] = None,
    ) -> SWEBenchPrediction:
        """
        SWE-bench prediction record를 만든다.
        """

        return SWEBenchPrediction(
            instance_id=instance_id,
            model_name_or_path=model_name_or_path or args.model_name,
            model_patch=model_patch,
        )

    def write_predictions(
        self,
        predictions: List[SWEBenchPrediction],
        output_path: Optional[str] = None,
    ) -> Path:
        """
        SWE-bench official harness가 읽는 predictions jsonl 파일을 저장한다.

        형식:
        {
          "instance_id": "...",
          "model_name_or_path": "...",
          "model_patch": "diff --git ..."
        }
        """

        path = Path(output_path) if output_path else self.predictions_path
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for prediction in predictions:
                f.write(json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n")

        logging.info(f"Saved SWE-bench predictions to: {path}")
        return path

    def write_single_prediction(
        self,
        instance_id: str,
        repo_dir: str,
        model_name_or_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Path:
        """
        repo_dir에서 patch를 뽑고 single prediction jsonl을 저장한다.
        """

        patch = self.extract_patch(repo_dir)

        prediction = self.build_prediction(
            instance_id=instance_id,
            model_patch=patch,
            model_name_or_path=model_name_or_path,
        )

        return self.write_predictions(
            predictions=[prediction],
            output_path=output_path,
        )

    def build_harness_command(
        self,
        predictions_path: Optional[str] = None,
        instance_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        SWE-bench official evaluation harness 실행 command를 만든다.
        """

        path = str(predictions_path or self.predictions_path)

        command = [
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            self.dataset_name,
            "--split",
            self.split,
            "--predictions_path",
            path,
            "--max_workers",
            str(self.max_workers),
            "--run_id",
            self.run_id,
            "--cache_level",
            self.cache_level,
        ]

        if self.clean:
            command.extend(["--clean", "True"])

        if self.modal:
            command.extend(["--modal", "true"])

        if instance_ids:
            command.append("--instance_ids")
            command.extend(instance_ids)

        return command

    def run_harness(
        self,
        predictions_path: Optional[str] = None,
        instance_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        official SWE-bench harness를 실행한다.
        """

        if self.dry_run:
            logging.info("SWE-bench dry run enabled. Skipping harness execution.")
            return {
                "dry_run": True,
                "predictions_path": str(predictions_path or self.predictions_path),
            }

        command = self.build_harness_command(
            predictions_path=predictions_path,
            instance_ids=instance_ids,
        )

        logging.info("Running SWE-bench harness:")
        logging.info(" ".join(command))

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        result = {
            "command": command,
            "return_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

        if completed.returncode != 0:
            logging.error("SWE-bench harness failed.")
            logging.error(completed.stderr)
            result["success"] = False
            return result

        result["success"] = True
        result["parsed_results"] = self.load_latest_results()
        return result

    def load_latest_results(self) -> Dict[str, Any]:
        """
        evaluation_results 아래에서 run_id가 포함된 가장 최근 results.json / instance_results.jsonl을 찾는다.
        SWE-bench 버전에 따라 경로가 조금 다를 수 있어서 탐색 방식으로 구현한다.
        """

        if not self.results_dir.exists():
            logging.warning(f"SWE-bench results dir does not exist: {self.results_dir}")
            return {}

        candidate_dirs = [
            path
            for path in self.results_dir.rglob("*")
            if path.is_dir() and self.run_id in str(path)
        ]

        search_roots = candidate_dirs if candidate_dirs else [self.results_dir]

        results_json = None
        instance_results_jsonl = None

        for root in search_roots:
            for path in root.rglob("results.json"):
                if results_json is None or path.stat().st_mtime > results_json.stat().st_mtime:
                    results_json = path

            for path in root.rglob("instance_results.jsonl"):
                if instance_results_jsonl is None or path.stat().st_mtime > instance_results_jsonl.stat().st_mtime:
                    instance_results_jsonl = path

        parsed: Dict[str, Any] = {}

        if results_json and results_json.exists():
            with results_json.open("r", encoding="utf-8") as f:
                parsed["results"] = json.load(f)
            parsed["results_path"] = str(results_json)

        if instance_results_jsonl and instance_results_jsonl.exists():
            instance_results = []
            with instance_results_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        instance_results.append(json.loads(line))
            parsed["instance_results"] = instance_results
            parsed["instance_results_path"] = str(instance_results_jsonl)

        return parsed

    def evaluate_repo_patch(
        self,
        instance_id: str,
        repo_dir: str,
        model_name_or_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        repo_dir의 git diff를 SWE-bench prediction으로 만들고, 해당 instance만 평가한다.
        """

        predictions_path = self.write_single_prediction(
            instance_id=instance_id,
            repo_dir=repo_dir,
            model_name_or_path=model_name_or_path,
        )

        return self.run_harness(
            predictions_path=str(predictions_path),
            instance_ids=[instance_id],
        )


def build_swebench_evaluator() -> SWEBenchEvaluator:
    return SWEBenchEvaluator(
        dataset_name=args.swebench_dataset_name,
        split=args.swebench_split,
        predictions_path=args.swebench_predictions_path,
        run_id=args.swebench_run_id,
        max_workers=args.swebench_max_workers,
        cache_level=args.swebench_cache_level,
        clean=args.swebench_clean,
        results_dir=args.swebench_results_dir,
        modal=args.swebench_modal,
        dry_run=args.swebench_dry_run,
    )


if __name__ == "__main__":
    repo_dir = args.swebench_repo_dir or args.executor_cwd

    evaluator = build_swebench_evaluator()

    prediction_path = evaluator.write_single_prediction(
        instance_id=args.rtv_test_problem if "__" in args.rtv_test_problem else "dummy__dummy-0",
        repo_dir=repo_dir,
        model_name_or_path=args.model_name,
    )

    print(f"Prediction file written to: {prediction_path}")

    if not args.swebench_dry_run:
        result = evaluator.run_harness(
            predictions_path=str(prediction_path),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))