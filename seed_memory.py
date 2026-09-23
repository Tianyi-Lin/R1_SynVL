import hashlib
from pathlib import Path

from configuration import connection_value, public_config
from storage import PROJECT_ROOT, digest, project_path, read_json, save_json


def build_manifest(config, seeds):
    modules = (
        "pipeline.py", "workflow.py", "configuration.py", "backend.py", "model_api.py", "image_api.py",
        "evaluation.py", "seed_memory.py", "seed_data.py", "storage.py",
    )
    memory_directory = config.get("seed_memory_dir", "seed_memory")
    lockfile = PROJECT_ROOT / "uv.lock"
    return {
        "config": public_config(config), "seeds": seeds,
        "source_hashes": {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in modules},
        "dependency_lock_sha256": hashlib.sha256(lockfile.read_bytes()).hexdigest() if lockfile.exists() else None,
        "image_hashes": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for seed in seeds for path in seed["images"]},
        "endpoints": {model["name"]: connection_value(config, model, "base_url") for model in config["models"]},
        "image_endpoint": connection_value({}, config["image_generation"], "base_url"),
        "seed_memory_dir": str(project_path(memory_directory)) if memory_directory else None,
    }


class SeedMemory:
    def __init__(self, config, seed, output, manifest):
        # 以题目文字和有序图片内容识别同题；seed_id 仅用于来源追踪，不限制记忆复用。
        identity = {"question": seed["question"],
                    "image_hashes": [manifest["image_hashes"][path] for path in seed["images"]]}
        runtime_fields = {"questions_per_seed", "max_iterations", "target_size", "seed_memory_dir", "request_timeout", "json_attempts", "dataset"}
        profile = {"config": public_config({key: value for key, value in config.items() if key not in runtime_fields}),
                   "endpoints": manifest["endpoints"], "image_endpoint": manifest["image_endpoint"]}
        self.seed_key = digest(identity)
        self.profile_key = digest(profile)
        self.seed_id = seed["seed_id"]
        entry_key = digest({"seed_key": self.seed_key, "seed_id": self.seed_id})
        root = Path(manifest["seed_memory_dir"]) if manifest["seed_memory_dir"] else output / "seed_memory"
        self.directory = root / self.seed_key / self.profile_key
        self.local_directory = output / "seed_states" / entry_key
        self.run_id = digest({"output": str(output), "manifest": manifest, "entry_key": entry_key})
        self.empty = {"schema_version": 2, "seed": identity, "seed_ids": [], "profile_key": self.profile_key,
                      "revision": None, "previous_revision": None, "rationales": {},
                      "previous_questions": {model["name"]: [] for model in config["models"]},
                      "optimized_context": "", "reflections": [], "generated_count": 0}

    def read_latest(self):
        pointer = self.directory / "latest.json"
        if not pointer.exists():
            return self.empty
        revision = read_json(pointer)["revision"]
        if not isinstance(revision, str) or len(revision) != 64 or any(character not in "0123456789abcdef" for character in revision):
            raise ValueError("Invalid seed memory revision")
        state = read_json(self.directory / "snapshots" / f"{revision}.json")
        if state.get("schema_version") != self.empty["schema_version"] or state["revision"] != revision or state["seed"] != self.empty["seed"] or state["profile_key"] != self.profile_key:
            raise ValueError("Seed memory identity or profile mismatch")
        return state

    def load_initial(self):
        initial = self.local_directory / "initial.json"
        if initial.exists():
            state = read_json(initial)
            if state.get("schema_version") != self.empty["schema_version"] or state["seed"] != self.empty["seed"] or state["profile_key"] != self.profile_key:
                raise ValueError("Initial seed memory belongs to different input or configuration")
            return state
        state = self.read_latest()
        save_json(initial, state)
        return state

    def ensure_current(self, state):
        if self.read_latest()["revision"] != state["revision"]:
            raise ValueError("Seed memory advanced in another run; resume that run or use a separate seed_memory_dir")

    def commit(self, state, record):
        if record["seed_id"] != self.seed_id:
            raise ValueError("Saved round belongs to a different source seed ID")
        revision = digest({"run_id": self.run_id, "iteration": record["iteration"]})
        seed_ids = list(state["seed_ids"])
        if self.seed_id not in seed_ids:
            seed_ids.append(self.seed_id)
        history = {name: list(questions) for name, questions in state["previous_questions"].items()}
        for candidate in record["candidates"]:
            history[candidate["generator"]].append(candidate["new_question_text"])
        reflections = list(state["reflections"])
        if record["reflection"] is not None:
            reflections.append({"run_id": self.run_id, "iteration": record["iteration"],
                                "reflection": record["reflection"]})
        updated = {**state, "revision": revision, "previous_revision": state["revision"],
                   "seed_ids": seed_ids, "source_seed_id": self.seed_id,
                   "source_output": str(self.local_directory.parent.parent), "source_iteration": record["iteration"],
                   "rationales": record["rationales"], "previous_questions": history,
                   "optimized_context": record["next_context"], "reflections": reflections,
                   "generated_count": state["generated_count"] + len(record["candidates"])}
        snapshot = self.directory / "snapshots" / f"{revision}.json"
        if snapshot.exists():
            if read_json(snapshot) != updated:
                raise ValueError("Seed memory snapshot differs from the saved round")
        else:
            self.ensure_current(state)
            save_json(snapshot, updated)
        if self.read_latest()["revision"] == state["revision"]:
            save_json(self.directory / "latest.json", {"revision": revision})
        save_json(self.local_directory / "latest.json", updated)
        return updated
