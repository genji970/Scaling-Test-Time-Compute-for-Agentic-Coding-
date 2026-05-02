import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from main.args import args


logging.basicConfig(level=logging.INFO)
logging.info("SWE-bench workspace module initialized.")


class SWEBenchWorkspace:
    """
    SWE-bench instance별 repo workspace를 준비하는 클래스.

    역할:
    1. sample["repo"] 기준으로 GitHub repo clone
    2. sample["base_commit"]으로 checkout
    3. 각 rollout마다 fresh workspace 복사
    4. executor_cwd가 해당 rollout workspace를 가리키게 만들 수 있도록 path 반환

    sample 예시:
    {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "base_commit": "...",
        "problem_statement": "..."
    }
    """

    def __init__(
        self,
        workspace_root: str,
        force_recreate: bool = False,
        keep_workspaces: bool = False,
        git_timeout: int = 600,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.force_recreate = force_recreate
        self.keep_workspaces = keep_workspaces
        self.git_timeout = git_timeout

        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def prepare_base_workspace(self, sample: Dict[str, Any]) -> Path:
        """
        instance의 base repo를 clone하고 base_commit으로 checkout한다.
        이 base workspace는 rollout별 workspace를 복사하기 위한 원본이다.
        """

        instance_id = self._require_field(sample, "instance_id")
        repo = self._require_field(sample, "repo")
        base_commit = self._require_field(sample, "base_commit")

        base_dir = self.get_base_dir(instance_id)

        if base_dir.exists() and self.force_recreate:
            logging.info(f"Removing existing base workspace: {base_dir}")
            shutil.rmtree(base_dir)

        if not base_dir.exists():
            repo_url = self.build_repo_url(repo)
            logging.info(f"Cloning repo {repo_url} into {base_dir}")

            base_dir.parent.mkdir(parents=True, exist_ok=True)

            self._run_command(
                command=["git", "clone", repo_url, str(base_dir)],
                cwd=self.workspace_root,
            )

        self.reset_to_base_commit(
            repo_dir=base_dir,
            base_commit=base_commit,
        )

        return base_dir

    def prepare_rollout_workspace(
        self,
        sample: Dict[str, Any],
        iteration: int,
        rollout_index: int,
    ) -> Path:
        """
        하나의 rollout을 위한 fresh workspace를 만든다.

        각 rollout은 서로 독립적이어야 하므로,
        base workspace를 복사해서 별도 디렉토리에서 실행한다.
        """

        base_dir = self.prepare_base_workspace(sample)
        instance_id = self._require_field(sample, "instance_id")

        rollout_dir = self.get_rollout_dir(
            instance_id=instance_id,
            iteration=iteration,
            rollout_index=rollout_index,
        )

        if rollout_dir.exists():
            logging.info(f"Removing existing rollout workspace: {rollout_dir}")
            shutil.rmtree(rollout_dir)

        logging.info(f"Copying base workspace to rollout workspace: {rollout_dir}")
        rollout_dir.parent.mkdir(parents=True, exist_ok=True)

        shutil.copytree(
            src=base_dir,
            dst=rollout_dir,
            symlinks=True,
        )

        base_commit = self._require_field(sample, "base_commit")

        self.reset_to_base_commit(
            repo_dir=rollout_dir,
            base_commit=base_commit,
        )

        return rollout_dir

    def reset_to_base_commit(self, repo_dir: Path, base_commit: str) -> None:
        """
        repo를 base_commit 상태로 강제 초기화한다.

        Some SWE-bench commits are not reachable from the default fetched refs.
        Therefore we first fetch normal refs, then explicitly try to fetch the
        target commit and pull refs before checkout.
        """

        logging.info(f"Resetting repo {repo_dir} to base commit {base_commit}")

        self._run_command(["git", "fetch", "--all", "--tags"], cwd=repo_dir)

        # Best-effort fetches for commits that are not reachable from default refs.
        self._run_command_allow_failure(
            ["git", "fetch", "origin", base_commit],
            cwd=repo_dir,
        )
        self._run_command_allow_failure(
            ["git", "fetch", "origin", "+refs/pull/*/head:refs/remotes/origin/pr/*"],
            cwd=repo_dir,
        )

        self._run_command(["git", "checkout", base_commit], cwd=repo_dir)
        self._run_command(["git", "reset", "--hard", base_commit], cwd=repo_dir)
        self._run_command(["git", "clean", "-fdx"], cwd=repo_dir)

    def get_base_dir(self, instance_id: str) -> Path:
        safe_instance_id = self._safe_name(instance_id)
        return self.workspace_root / safe_instance_id / "base"

    def get_rollout_dir(
        self,
        instance_id: str,
        iteration: int,
        rollout_index: int,
    ) -> Path:
        safe_instance_id = self._safe_name(instance_id)
        return (
            self.workspace_root
            / safe_instance_id
            / f"iter_{iteration}"
            / f"rollout_{rollout_index}"
        )

    def build_repo_url(self, repo: str) -> str:
        """
        SWE-bench sample의 repo field는 보통 'owner/name' 형태다.
        """
        if repo.startswith("http://") or repo.startswith("https://") or repo.endswith(".git"):
            return repo

        return f"https://github.com/{repo}.git"

    def cleanup_instance(self, instance_id: str) -> None:
        """
        keep_workspaces가 False일 때 instance workspace 전체를 삭제한다.
        """
        if self.keep_workspaces:
            return

        instance_dir = self.workspace_root / self._safe_name(instance_id)

        if instance_dir.exists():
            logging.info(f"Cleaning up SWE-bench workspace: {instance_dir}")
            shutil.rmtree(instance_dir)

    def _run_command_allow_failure(self, command: list[str], cwd: Path) -> subprocess.CompletedProcess:
        """
        Optional command runner.
        Failure is logged but does not stop workspace setup.
        Used for best-effort git fetch attempts.
        """

        logging.info(f"Running optional command in {cwd}: {' '.join(command)}")

        output_chunks = []

        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None

            for line in process.stdout:
                print(f"[git/optional] {line}", end="", flush=True)
                output_chunks.append(line)

            returncode = process.wait(timeout=self.git_timeout)

        except Exception as error:
            logging.warning(f"Optional command raised error: {error}")
            return subprocess.CompletedProcess(
                args=command,
                returncode=1,
                stdout="".join(output_chunks),
                stderr=str(error),
            )

        output = "".join(output_chunks)

        if returncode != 0:
            logging.warning(
                "Optional command failed. Continuing.\n"
                f"Command: {' '.join(command)}\n"
                f"OUTPUT:\n{output}"
            )

        return subprocess.CompletedProcess(
            args=command,
            returncode=returncode,
            stdout=output,
            stderr="",
        )

    def _run_command(self, command: list[str], cwd: Path) -> subprocess.CompletedProcess:
        logging.info(f"Running command in {cwd}: {' '.join(command)}")

        output_chunks = []

        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            try:
                assert process.stdout is not None

                for line in process.stdout:
                    print(f"[git/workspace] {line}", end="", flush=True)
                    output_chunks.append(line)

                returncode = process.wait(timeout=self.git_timeout)

            except subprocess.TimeoutExpired:
                process.kill()
                timeout_msg = f"\nCommand timed out after {self.git_timeout} seconds.\n"
                print(f"[git/workspace] {timeout_msg}", flush=True)
                output_chunks.append(timeout_msg)
                returncode = 124

        except FileNotFoundError as error:
            raise RuntimeError(f"Command not found: {command[0]}\n{error}") from error

        output = "".join(output_chunks)

        completed = subprocess.CompletedProcess(
            args=command,
            returncode=returncode,
            stdout=output,
            stderr="",
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"Command failed: {' '.join(command)}\n"
                f"CWD: {cwd}\n"
                f"Return code: {completed.returncode}\n"
                f"OUTPUT:\n{completed.stdout}"
            )

        return completed

    @staticmethod
    def _safe_name(name: str) -> str:
        return (
            name.replace("/", "__")
            .replace(":", "_")
            .replace(" ", "_")
        )

    @staticmethod
    def _require_field(sample: Dict[str, Any], field_name: str) -> str:
        value = sample.get(field_name)

        if value is None or value == "":
            raise ValueError(f"SWE-bench sample is missing required field: {field_name}")

        return str(value)


def build_swebench_workspace() -> SWEBenchWorkspace:
    return SWEBenchWorkspace(
        workspace_root=args.swebench_workspace_root,
        force_recreate=args.swebench_force_recreate_workspace,
        keep_workspaces=args.swebench_keep_workspaces,
        git_timeout=args.swebench_git_timeout,
    )


if __name__ == "__main__":
    from data_download.download import DataDownloader

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

    samples = downloader.normalize_dataset()

    if not samples:
        raise ValueError("No samples loaded.")

    sample = samples[args.pipeline_sample_index]

    workspace = build_swebench_workspace()

    rollout_dir = workspace.prepare_rollout_workspace(
        sample=sample,
        iteration=0,
        rollout_index=0,
    )

    print(f"Prepared rollout workspace: {rollout_dir}")