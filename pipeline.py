import argparse
import json
from pathlib import Path

from configuration import load_config, validate_config
from seed_data import load_dataset_seeds, load_seeds
from workflow import generation_batches, run_pipeline


def main():
    parser = argparse.ArgumentParser(description="R1-SyntheticVL CADS data synthesis; see README for release configuration")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--seeds", help="Legacy local JSON input; overrides the configured dataset")
    source.add_argument("--dataset", help="Hub dataset ID or local Parquet/JSON/JSONL file; defaults to dataset.path")
    parser.add_argument("--split", help="Dataset split; defaults to dataset.split")
    parser.add_argument("--images-dir")
    parser.add_argument("--output", default="outputs/default")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.seeds and args.split:
        parser.error("--split applies to dataset input, not legacy --seeds JSON")
    config = load_config(args.config)
    if args.dataset:
        config["dataset"]["path"] = args.dataset
    if args.split:
        config["dataset"]["split"] = args.split
    validate_config(config, live=not args.dry_run)
    seeds = load_seeds(args.seeds, args.images_dir, args.limit) if args.seeds else load_dataset_seeds(config["dataset"], args.images_dir, args.limit)
    if args.dry_run:
        batches = generation_batches(config)
        allocation = {model["name"]: 0 for model in config["models"]}
        for batch in batches:
            for sample_index, name in batch:
                allocation[name] += 1
        preview = {"seeds": len(seeds), "questions_per_seed": config["questions_per_seed"],
                   "collective_model_count": len(config["models"]), "generation_rounds": len(batches),
                   "allocation_per_seed": allocation, "candidate_upper_bound": len(seeds) * config["questions_per_seed"],
                   "target_size": config["target_size"], "model_api_calls": 0}
        if args.seeds:
            preview["network_calls"] = 0
        else:
            preview["dataset"] = {"path": config["dataset"]["path"], "split": config["dataset"]["split"]}
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return
    run_pipeline(config, seeds, args.output)


if __name__ == "__main__":
    main()
