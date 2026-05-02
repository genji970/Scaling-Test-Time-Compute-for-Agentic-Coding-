import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from main.args import args


logging.basicConfig(level=logging.INFO)
logging.info("Data downloader module initialized.")


class DataDownloader:
    """
    Hugging Face dataset을 다운로드하고,
    agentic coding pipeline에서 쓰기 쉬운 공통 schema로 정규화한다.

    논문 구조 기준:
    - SWE-Bench Verified 또는 Terminal-Bench 같은 benchmark sample을 불러온다.
    - 각 sample에서 problem_statement를 추출한다.
    - pipeline/agent가 이 problem_statement를 받아 rollout을 생성한다.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str = "",
        split: str = "test",
        cache_dir: str = "data_download/cache",
        max_samples: int = -1,
        problem_field: str = "problem_statement",
        instance_id_field: str = "instance_id",
        repo_field: str = "repo",
        base_commit_field: str = "base_commit",
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.cache_dir = cache_dir
        self.max_samples = max_samples
        self.problem_field = problem_field
        self.instance_id_field = instance_id_field
        self.repo_field = repo_field
        self.base_commit_field = base_commit_field

    def download(self):
        """
        Hugging Face dataset을 로드한다.
        dataset_config가 비어 있으면 config 없이 로드한다.
        """

        logging.info(f"Loading dataset: {self.dataset_name}")
        logging.info(f"Dataset config: {self.dataset_config or '[none]'}")
        logging.info(f"Split: {self.split}")

        if self.dataset_config:
            dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=self.split,
                cache_dir=self.cache_dir,
            )
        else:
            dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=self.cache_dir,
            )

        if self.max_samples is not None and self.max_samples > 0:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))

        logging.info(f"Loaded samples: {len(dataset)}")
        return dataset

    def normalize_sample(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        dataset마다 field 이름이 다를 수 있으므로,
        pipeline에서 공통으로 쓸 수 있는 형태로 바꾼다.
        """

        problem_statement = self._get_field(
            sample=sample,
            field_name=self.problem_field,
            default="",
        )

        instance_id = self._get_field(
            sample=sample,
            field_name=self.instance_id_field,
            default=str(index),
        )

        repo = self._get_field(
            sample=sample,
            field_name=self.repo_field,
            default="",
        )

        base_commit = self._get_field(
            sample=sample,
            field_name=self.base_commit_field,
            default="",
        )

        return {
            "index": index,
            "instance_id": instance_id,
            "problem_statement": problem_statement,
            "repo": repo,
            "base_commit": base_commit,
            "raw": dict(sample),
        }

    def normalize_dataset(self) -> List[Dict[str, Any]]:
        """
        전체 dataset을 list[dict] 형태의 normalized samples로 변환한다.
        """

        dataset = self.download()
        normalized_samples = []

        for index, sample in enumerate(dataset):
            normalized = self.normalize_sample(
                sample=sample,
                index=index,
            )
            normalized_samples.append(normalized)

        return normalized_samples

    def save_jsonl(self, samples: List[Dict[str, Any]], output_path: str) -> None:
        """
        normalized samples를 jsonl로 저장한다.
        """

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logging.info(f"Saved normalized dataset to: {output_path}")

    def run(self) -> List[Dict[str, Any]]:
        """
        다운로드 → 정규화 → 저장까지 수행한다.
        """

        samples = self.normalize_dataset()
        self.save_jsonl(samples, args.save_dataset_path)
        return samples

    @staticmethod
    def _get_field(
        sample: Dict[str, Any],
        field_name: str,
        default: Optional[Any] = None,
    ) -> Any:
        """
        field가 없을 때도 안전하게 가져오기 위한 helper.
        """

        if field_name in sample:
            return sample[field_name]

        logging.warning(f"Field not found: {field_name}")
        return default


if __name__ == "__main__":
    downloader = DataDownloader(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.train_split,
        cache_dir=args.data_cache_dir,
        max_samples=args.max_dataset_samples,
        problem_field=args.problem_field,
        instance_id_field=args.instance_id_field,
        repo_field=args.repo_field,
        base_commit_field=args.base_commit_field,
    )

    samples = downloader.run()

    print(f"Loaded normalized samples: {len(samples)}")

    if samples:
        print("First sample:")
        print(json.dumps(samples[0], indent=2, ensure_ascii=False))