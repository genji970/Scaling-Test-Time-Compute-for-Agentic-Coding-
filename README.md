# Scaling-Test-Time-Compute-for-Agentic-Coding-
paper implementation of Meta Ai

# gitl clone 

# cd your/path/

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
