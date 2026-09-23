import argparse
from collections import Counter
from pathlib import Path

from evaluation import consensus
from storage import PROJECT_ROOT, read_json, save_json


def audit(root):
    root = Path(root)
    generated = read_json(root / "generated_results.json")
    verification = read_json(root / "verify.json")
    cache = read_json(root / "verify_multi_answer_cache.json")
    pass_ids = set(map(str, read_json(root / "verify_pass_ids.json")))
    records = verification.values() if isinstance(verification, dict) else verification
    records = {str(item["idx"]): item for item in records}
    counts = Counter()
    groups = Counter()
    unknown = Counter()
    reconstructed = {}
    for seed_id, record in records.items():
        answers = record.get("model_answers", {})
        names = sorted(answers)
        groups[" + ".join(names)] += 1
        entry = cache.get(seed_id, {})
        try:
            if len(names) != 3:
                raise ValueError("unexpected_legacy_judge_count")
            if any(not isinstance(answer, dict) or not answer.get("raw") or answer["raw"].startswith("[ERROR:") for answer in answers.values()):
                raise ValueError("incomplete_solver_response")
            reference = str(generated.get(seed_id, {}).get("new_question_answer", "")).strip()
            if not reference or str(entry.get("new_question_answer", "")).strip() != reference:
                raise ValueError("missing_or_stale_reference")
            if entry.get("reason") == "strict_match":
                boxed = entry.get("boxed_answers", {})
                if set(boxed) != set(names) or any(str(answer).strip() != reference for answer in boxed.values()):
                    raise ValueError("invalid_strict_match_cache")
                judgements = {name: True for name in names}
            elif entry.get("reason") == "gpt_judged":
                judgements = entry.get("gpt_judgement", {})
            else:
                raise ValueError("missing_or_failed_equivalence_judgement")
            decision = consensus(judgements, names)
            counts[str(decision["consensus_score"])] += 1
            reconstructed[seed_id] = {**decision, "models": names, "provenance": "legacy_cache_unrevalidated"}
        except ValueError as error:
            reason = str(error)
            unknown[reason] += 1
            reconstructed[seed_id] = {"status": "unknown", "reason": reason, "models": names}
    qualified = {seed_id for seed_id, item in reconstructed.items() if item.get("qualified")}
    summary = {
        "generated": len(generated), "verification": len(records), "cache": len(cache),
        "legacy_pass_ids": len(pass_ids), "model_groups": dict(groups),
        "consensus_distribution_K3": dict(sorted(counts.items())),
        "adversarial_K3": sum(item.get("adversarial", False) for item in reconstructed.values()),
        "qualified_K3": len(qualified), "unknown": dict(unknown),
        "generated_without_verification": len(set(generated) - set(records)),
        "legacy_pass_not_reconstructed": len(pass_ids - qualified),
        "reconstructed_not_legacy_pass": len(qualified - pass_ids),
        "paper_K4_reproducible": False,
        "note": "Uses existing cached equivalence decisions; no new model judgement or visual validation. K=3 does not establish K=4 consensus.",
    }
    return {"summary": summary, "records": reconstructed}


def main():
    parser = argparse.ArgumentParser(description="Offline audit of existing three-model CADS-like results")
    parser.add_argument("--root", default=str(PROJECT_ROOT.parent / "R1-ShareVL-52K" / "syn_data"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "legacy_audit.json"))
    args = parser.parse_args()
    result = audit(args.root)
    save_json(args.output, result)
    import json
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
