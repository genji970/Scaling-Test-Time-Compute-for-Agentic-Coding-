# Scaling-Test-Time-Compute-for-Agentic-Coding-

paper implementation of Meta Ai

```python
This paper was released on April 16, 2026.

This paper argues that the main bottleneck in test-time scaling for agentic coding is not simply generating more attempts, but representing, selecting, and reusing prior agent experience effectively. It converts long rollout trajectories into compact structured summaries that capture key hypotheses, progress, and failure modes. Using these summaries, it combines Recursive Tournament Voting (RTV) for parallel selection with Parallel-Distill-Refine (PDR) for sequential improvement. The method consistently improves frontier coding agents on SWE-Bench Verified and Terminal-Bench v2.0. Overall, the paper frames long-horizon agent scaling as a problem of representation, selection, and reuse.
```
`As far as I know, there is no public implementation of this paper yet.`

## Terminal Code
`git clone https://github.com/genji970/harness-inference-scale-agent`

`cd your/path/harness-inference-scale-agent`

`change gemini_api_key value in below code, GEMINI_API_KEY= & gemini_api_key= `

## Experiment Run

`pip install -r requirements.txt`

`litellm --model gemini/gemini-3.1-pro --port 4000`

And make one more terminal. And run below code in new terminal.

```python
source .env

MSWEA_MODEL_NAME="gemini/gemini-3-pro-preview" \
GEMINI_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxx" \
python experiment/run.py \
  --provider "gemini" \
  --gemini_api_key "xxxxxxxxxxxxxxxxxxxxxxxx" \
  --dataset_name "princeton-nlp/SWE-bench_Verified" \
  --train_split "test" \
  --max_dataset_samples 1 \
  --start_sample_index 0 \
  --end_sample_index 1 \
  --num_rollouts 2 \
  --num_iterations 2 \
  --top_k 1 \
  --group_size 2 \
  --vote_count 1 \
  --max_steps_per_rollout 10 \
  --agent_max_steps 1 \
  --pdr_max_summary_chars 4000 \
  --pdr_max_context_chars 16000 \
  --mini_swe_agent_extra_args="--yolo" \
  --swebench_prepare_workspace \
  --swebench_enabled \
  --swebench_dry_run \
  --swebench_max_workers 1 \
  --swebench_cache_level env \
  --continue_on_error \
  --experiment_output_path "outputs/pdr_rtv_results.jsonl" \
  --experiment_summary_path "outputs/pdr_rtv_summary.json"
```

## Installation & Run for Real benchmark test
```python
# docker install & start
apt-get update
apt-get install -y docker.io

service docker start

docker version
docker info

# SWE-bench harness install
python -m pip install --upgrade pip
python -m pip install swebench

#Public SWE-bench eval run
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --split test \
  --predictions_path outputs/swebench_predictions.jsonl \
  --max_workers 1 \
  --run_id gemini_3_pro_pdr_rtv_eval \
  --cache_level env
```



## Implementation 

Paper introduces gemini-3.1, opus, etc and two benchmark dataset for evaluation.
But I only coded for gemini-3.1-pro and for SWEbenchmark only.

# paper link
`https://arxiv.org/abs/2604.16529v1`


## Citatiton
@misc{kim2026scalingtesttimecomputeagentic,
      title={Scaling Test-Time Compute for Agentic Coding}, 
      author={Joongwon Kim and Wannan Yang and Kelvin Niu and Hongming Zhang and Yun Zhu and Eryk Helenowski and Ruan Silva and Zhengxing Chen and Srinivasan Iyer and Manzil Zaheer and Daniel Fried and Hannaneh Hajishirzi and Sanjeev Arora and Gabriel Synnaeve and Ruslan Salakhutdinov and Anirudh Goyal},
      year={2026},
      eprint={2604.16529},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2604.16529}, 
}
