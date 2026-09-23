#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
download_dataset=false
dataset_options=false
config_path="$project_dir/config.yaml"
dataset_args=()
limit=2

usage() {
    cat <<'HELP'
Usage: bash cads/setup.sh [--download-dataset] [dataset options]

Installs locked dependencies into cads/.venv. No model API calls.
If uv is unavailable, requires python3 with pip to install uv locally.

  --download-dataset  Download/cache the configured dataset and check sample rows
  --config PATH       Configuration file (default: cads/config.yaml)
  --dataset ID        Override the Hub dataset ID or local dataset file
  --split NAME        Override the dataset split (default: configured split)
  --limit N           Number of rows to validate (default: 2, not a download limit)
  --help              Show this message without installing anything

Dataset options require --download-dataset. Relative input paths are resolved
from the calling directory. Dataset/cache files stay under cads/dataset/ by
default; downloads never start model generation or evaluation.
HELP
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --download-dataset)
            download_dataset=true
            shift
            ;;
        --config|--dataset|--split|--limit)
            if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            dataset_options=true
            case "$1" in
                --config) config_path="$2" ;;
                --dataset|--split) dataset_args+=("$1" "$2") ;;
                --limit) limit="$2" ;;
            esac
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$dataset_options" == true && "$download_dataset" != true ]]; then
    printf 'Dataset options require --download-dataset.\n' >&2
    exit 2
fi
if [[ ! "$limit" =~ ^[0-9]+$ || ! "$limit" =~ [1-9] ]]; then
    printf -- '--limit must be a positive integer.\n' >&2
    exit 2
fi
if [[ "$download_dataset" == true && ! -f "$config_path" ]]; then
    printf 'Missing configuration: %s\n' "$config_path" >&2
    exit 2
fi

export UV_PROJECT_ENVIRONMENT="$project_dir/.venv"
export UV_CACHE_DIR="$project_dir/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$project_dir/.python"
export TMPDIR="$project_dir/.tmp"
export HF_HOME="$project_dir/dataset/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

if [[ -x "$project_dir/.tools/uv/bin/uv" ]]; then
    uv_command="$project_dir/.tools/uv/bin/uv"
elif command -v uv >/dev/null 2>&1; then
    uv_command="$(command -v uv)"
else
    if ! command -v python3 >/dev/null 2>&1 || ! python3 -m pip --version >/dev/null 2>&1; then
        printf 'Install uv, or provide python3 with pip, then rerun this script.\n' >&2
        exit 1
    fi
    python3 -m pip install --disable-pip-version-check --cache-dir "$project_dir/.cache/pip" \
        --target "$project_dir/.tools/uv" 'uv==0.12.17'
    uv_command="$project_dir/.tools/uv/bin/uv"
fi

"$uv_command" sync --project "$project_dir" --locked
"$project_dir/.venv/bin/python" -c 'from importlib.metadata import version; print("Installed: " + ", ".join(f"{name}=={version(name)}" for name in ("datasets", "google-genai", "openai", "Pillow", "PyYAML")))'
printf 'Environment ready: %s/.venv\n' "$project_dir"

if [[ "$download_dataset" == true ]]; then
    printf 'Downloading/caching dataset and validating %s rows; no model API calls.\n' "$limit"
    "$project_dir/.venv/bin/python" "$project_dir/pipeline.py" --config "$config_path" \
        --dry-run --limit "$limit" ${dataset_args[@]+"${dataset_args[@]}"}
fi
