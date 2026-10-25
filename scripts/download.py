"""Fetch the raw datasets and candidate base models into the project tree."""
from huggingface_hub import snapshot_download

DATASETS = ["westenfelder/NL2SH-ALFA", "westenfelder/InterCode-Corrections", "jiacheng-ye/nl2bash"]
MODELS = [
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3.5-0.8B",
    "ISB369/shellminator-qwen05b-dpo-selfplay",
    "dhwanichande29/nl-to-bash",
    "sahellx/kitty-bash-llm",
]

for repo in DATASETS:
    snapshot_download(repo, repo_type="dataset", local_dir=f"data/raw/{repo.split('/')[1]}")
    print("dataset", repo)
for repo in MODELS:
    snapshot_download(repo, local_dir=f"models/base/{repo.split('/')[1]}")
    print("model", repo)
