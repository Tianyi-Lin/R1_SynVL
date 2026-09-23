from backend import Backend
from configuration import validate_config
from evaluation import judge_candidate
from seed_memory import SeedMemory, build_manifest
from storage import digest, project_path, read_json, require_text, save_json


STRATEGIES = (
    "Numerical & Parameter Variation",
    "Logic Reversion",
    "Auxiliary Extension",
    "Isomorphic Scenario Transfer",
)


def generation_batches(config):
    names = [model["name"] for model in config["models"]]
    count = config["questions_per_seed"]
    cycles_per_round = (count + len(names) * config["max_iterations"] - 1) // (len(names) * config["max_iterations"])
    batch_size = max(1, cycles_per_round) * len(names)
    return [
        [(index + 1, names[index % len(names)]) for index in range(start, min(start + batch_size, count))]
        for start in range(0, count, batch_size)
    ]


def validate_candidate(value):
    require_text(value, ("domain", "core_concepts", "strategy_selected", "strategy_detail", "new_question_text", "new_question_answer", "new_solution_text"))
    if value["strategy_selected"] not in STRATEGIES:
        raise ValueError("Unknown meta strategy")
    if value.get("task_type") not in ("MCQ", "Open-ended"):
        raise ValueError("Unknown task type")
    if value["task_type"] == "MCQ" and value["new_question_answer"] not in ("A", "B", "C", "D"):
        raise ValueError("MCQ answer must be a single option letter")
    return value


def analyze_seed(backend, name, seed):
    return backend.ask(
        name, "Analyze the seed's domain, core concepts, constraints and valid solution rationale. "
        "For a task description, identify the concepts and reasoning that new tasks should exercise. "
        "Return {domain: string, core_concepts: string, rationale: string}.",
        {"seed": seed["question"]}, seed["images"], ("domain", "core_concepts", "rationale"),
    )


def generate_candidate(backend, name, seed, rationale, context, previous_questions, iteration, sample_index):
    return backend.ask(
        name, "Generate one new, solvable visual problem using your own rationale analysis. "
        "Derive a concrete strategy for this seed from exactly one supplied meta strategy. "
        "Use the optimized context to target reasoning weaknesses while preserving validity. "
        "Avoid repeating previous questions. Make visual evidence necessary. "
        "Return one JSON object with exactly these top-level fields: domain, core_concepts, task_type, "
        "strategy_selected, strategy_detail, new_question_text, new_question_answer, new_solution_text. "
        "Every field value must be a nonempty string; core_concepts must be a single string, not an array. "
        "Do not nest the required fields, copy input-only fields or add an all_strings field. "
        "task_type must be MCQ or Open-ended; strategy_selected must be an exact supplied meta strategy name. "
        "MCQ new_question_text must contain all four options A–D and new_question_answer must be the letter only. "
        "Open-ended answers must be minimal; specify required units in the question.",
        {"seed_id": seed["seed_id"], "seed": seed["question"], "rationale": rationale,
         "meta_strategies": STRATEGIES, "optimized_context": context,
         "previous_questions": list(previous_questions), "iteration": iteration,
         "sample_index": sample_index},
        seed["images"], validator=validate_candidate,
    )


def write_image_prompt(backend, name, candidate):
    return backend.ask(
        name, "Write a precise executable drawing prompt for the new problem: spatial layout, "
        "object attributes, data values, geometric relationships and readable labels. "
        "Render evidence needed to solve it, without the final answer or solution annotations. "
        "Return {image_generation_prompt: string}.",
        candidate, fields=("image_generation_prompt",),
    )


def run_pipeline(config, seeds, output, backend=None):
    validate_config(config, live=backend is None)
    output = project_path(output)
    manifest = build_manifest(config, seeds)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Output belongs to a different run configuration/input/code; use a new output directory")
    save_json(manifest_path, manifest)
    backend = backend or Backend(config, output)
    names = [model["name"] for model in config["models"]]
    accepted = {}
    adversarial = {}
    all_candidates = {}
    batches = generation_batches(config)
    for seed in seeds:
        if config["target_size"] is not None and len(accepted) >= config["target_size"]:
            break
        memory = SeedMemory(config, seed, output, manifest)
        state = memory.load_initial()
        sample_offset = state["generated_count"]
        context = state["optimized_context"]
        previous_questions = state["previous_questions"]
        for iteration, batch in enumerate(batches, start=1):
            round_key = digest({"seed": seed["seed_id"], "iteration": iteration})
            round_path = output / "rounds" / f"{round_key}.json"
            if round_path.exists():
                record = read_json(round_path)
            else:
                memory.ensure_current(state)
                # 同题、同配置下复用各模型已保存的初始分析，避免每批或更换 seed ID 后重复分析。
                # 每个模型只使用自己的分析；没有可复用记录时才分别请求分析。
                rationales = state["rationales"] or {
                    name: analyze_seed(backend, name, seed) for name in names
                }
                candidates = []
                round_previous = {name: list(questions) for name, questions in previous_questions.items()}
                for local_index, name in batch:
                    sample_index = sample_offset + local_index
                    candidate = generate_candidate(
                        backend, name, seed, rationales[name], context,
                        round_previous[name], iteration, sample_index,
                    )
                    visual = write_image_prompt(backend, name, candidate)
                    candidate = {**candidate, **visual}
                    candidate["generated_image_path"] = backend.image(visual["image_generation_prompt"])
                    candidate.update({"idx": f"{seed['seed_id']}:{sample_index}:{name}", "seed_id": seed["seed_id"],
                                      "iteration": iteration, "sample_index": sample_index,
                                      "generator": name, "context_used": context})
                    candidate.update(judge_candidate(backend, candidate, names, config))
                    candidates.append(candidate)
                    round_previous[name].append(candidate["new_question_text"])
                hard = [candidate for candidate in candidates if candidate["adversarial"]]
                next_context = context
                reflection = None
                if hard:
                    reflection = backend.ask(
                        config["reflection_model"],
                        "Reflect on disagreements between judges for these solvable adversarial samples. "
                        "Distill success/error patterns, then optimize generation guidance for future problems. "
                        "Target challenging reasoning, never ambiguity, missing evidence or incorrect labels. "
                        "Return {reflection: string, optimized_context: string}. Keep optimized_context under 1500 words.",
                        {"previous_context": context, "adversarial_samples": hard},
                        [candidate["generated_image_path"] for candidate in hard],
                        ("reflection", "optimized_context"),
                    )
                    next_context = reflection["optimized_context"]
                record = {"seed_id": seed["seed_id"], "iteration": iteration, "rationales": rationales,
                          "candidates": candidates, "reflection": reflection, "next_context": next_context}
                save_json(round_path, record)
            state = memory.commit(state, record)
            context = state["optimized_context"]
            previous_questions = state["previous_questions"]
            for candidate in record["candidates"]:
                all_candidates[candidate["idx"]] = candidate
                if candidate["qualified"]:
                    accepted[candidate["idx"]] = candidate
                if candidate["adversarial"]:
                    adversarial[candidate["idx"]] = candidate
            save_json(output / "generated_results.json", accepted)
            save_json(output / "adversarial_results.json", adversarial)
            save_json(output / "all_candidates.json", all_candidates)
            save_json(output / "training_data.json", [
                {"idx": candidate["idx"], "images": [candidate["generated_image_path"]],
                 "problem": "<image>" + candidate["new_question_text"], "answer": candidate["new_question_answer"]}
                for candidate in accepted.values()
            ])
            print(f"seed={seed['seed_id']} iteration={iteration}/{len(batches)} generated={len(all_candidates)} "
                  f"accepted={len(accepted)} adversarial={len(adversarial)}", flush=True)
