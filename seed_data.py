import hashlib
import io
import os
from itertools import islice
from pathlib import Path


from storage import PROJECT_ROOT, read_json


# Default input follows EasyR1's recommended problem/answer/images dataset format.
# Reference: https://github.com/hiyouga/EasyR1#custom-dataset and hiyouga/geometry3k.
DEFAULT_DATASET = {
    "path": "hiyouga/geometry3k",
    "name": None,
    "split": "train",
    "revision": "main",
    "prompt_key": "problem",
    "answer_key": "answer",
    "image_key": "images",
    "id_key": None,
    "image_cache_dir": str(PROJECT_ROOT / "dataset" / "seed_images"),
}


def validate_dataset_config(config):
    if not isinstance(config, dict):
        raise ValueError("dataset must be a mapping")
    for field in ("path", "split", "prompt_key", "answer_key", "image_key", "image_cache_dir"):
        if not isinstance(config.get(field), str) or not config[field].strip():
            raise ValueError(f"dataset.{field} must be a nonempty string")
    for field in ("name", "revision", "id_key"):
        if config.get(field) is not None and (not isinstance(config[field], str) or not config[field].strip()):
            raise ValueError(f"dataset.{field} must be a nonempty string or null")


def resolve_image_path(value, base_dir, images_dir):
    path = Path(value)
    candidates = []
    if images_dir:
        if not path.is_absolute():
            candidates.append(Path(images_dir) / path)
        candidates.append(Path(images_dir) / path.name)
    candidates.extend([base_dir / path, path])
    resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if resolved is None:
        raise FileNotFoundError(f"Missing dataset image: {value}")
    return resolved


def materialize_image(value, cache_dir, base_dir, images_dir):
    from PIL import Image

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            value = value["bytes"]
        elif value.get("path"):
            value = value["path"]
        else:
            raise ValueError("Image mapping must contain bytes or a local path")
    if isinstance(value, Image.Image):
        if getattr(value, "n_frames", 1) != 1:
            raise ValueError("Multi-frame images are not supported as seed images")
        source = value.copy()
    elif isinstance(value, (bytes, bytearray, memoryview)):
        source = Image.open(io.BytesIO(bytes(value)))
    elif isinstance(value, (str, Path)):
        source = Image.open(resolve_image_path(value, base_dir, images_dir))
    else:
        raise ValueError("Images must be PIL images, bytes, local paths, or {bytes, path} mappings")
    with source:
        source.load()
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("Multi-frame images are not supported as seed images")
        mode = "RGBA" if source.mode in ("RGBA", "LA", "P") else "RGB"
        with source.convert(mode) as normalized:
            buffer = io.BytesIO()
            normalized.save(buffer, format="PNG")
    encoded = buffer.getvalue()
    checksum = hashlib.sha256(encoded).hexdigest()
    path = Path(cache_dir) / f"{checksum}.png"
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Corrupt seed image cache: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.png")
        temporary.write_bytes(encoded)
        temporary.replace(path)
    return str(path.resolve())


def load_dataset_seeds(config, images_dir=None, limit=None):
    """Read EasyR1-style rows; archive source answers without sending them to generators or judges."""
    validate_dataset_config(config)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "dataset" / "huggingface"))
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ValueError("Hugging Face/Parquet input requires datasets; run uv sync --locked in cads/") from error

    path = Path(config["path"])
    if path.is_file() or path.suffix.lower() in (".parquet", ".json", ".jsonl"):
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        file_type = path.suffix.lower().lstrip(".").replace("jsonl", "json")
        if file_type not in ("parquet", "json"):
            raise ValueError("Local dataset input supports Parquet, JSON and JSONL files")
        dataset = load_dataset(file_type, data_files={config["split"]: str(path.resolve())}, split=config["split"])
        source_path = str(path.resolve())
        base_dir = path.resolve().parent
        source_name = None
        source_revision = None
    else:
        if path.is_dir() or config["path"].startswith(("/", "./", "../", "http://", "https://")):
            raise ValueError("dataset.path must be a Hub dataset ID or a local Parquet/JSON/JSONL file")
        options = {"split": config["split"]}
        for field in ("name", "revision"):
            if config.get(field) is not None:
                options[field] = config[field]
        dataset = load_dataset(config["path"], **options)
        source_path = config["path"]
        base_dir = Path.cwd()
        source_name = config.get("name")
        source_revision = config.get("revision")

    required = [config[field] for field in ("prompt_key", "answer_key", "image_key")]
    if config.get("id_key"):
        required.append(config["id_key"])
    missing = set(required) - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset must follow the configured EasyR1 schema; missing columns: {sorted(missing)}")
    seeds = []
    seen = set()
    for row_index, item in enumerate(islice(dataset, limit)):
        row_id = item[config["id_key"]] if config.get("id_key") else row_index
        if type(row_id) not in (str, int) or not str(row_id).strip():
            raise ValueError(f"Invalid dataset seed ID at row {row_index}")
        seed_id = f"{source_path}:{source_name or 'default'}:{config['split']}:{row_id}"
        if seed_id in seen:
            raise ValueError(f"Duplicate seed ID: {seed_id}")
        seen.add(seed_id)
        problem = item[config["prompt_key"]]
        answer = item[config["answer_key"]]
        if not isinstance(problem, str) or not problem.replace("<image>", "").strip():
            raise ValueError(f"Missing nonempty problem text: {seed_id}")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"EasyR1 answer must be a nonempty string: {seed_id}")
        declared = item[config["image_key"]]
        if not isinstance(declared, (list, tuple)):
            raise ValueError(f"EasyR1 images must be a list: {seed_id}")
        if problem.count("<image>") != len(declared):
            raise ValueError(f"Each image needs a matching <image> placeholder: {seed_id}")
        images = [materialize_image(value, config["image_cache_dir"], base_dir, images_dir) for value in declared]
        seeds.append({"seed_id": seed_id, "question": problem.replace("<image>", "").strip(),
                      "images": images, "source_answer": answer,
                      "source": {"path": source_path, "name": source_name, "split": config["split"],
                                 "revision": source_revision, "row_index": row_index,
                                 "dataset_fingerprint": getattr(dataset, "_fingerprint", None)}})
    if not seeds:
        raise ValueError("No seeds selected")
    return seeds


def load_seeds(path, images_dir, limit):
    data = read_json(path)
    entries = data.items() if isinstance(data, dict) else enumerate(data)
    seeds = []
    seen = set()
    for key, item in entries:
        seed_id = str(item.get("idx", key))
        if seed_id in seen:
            raise ValueError(f"Duplicate seed ID: {seed_id}")
        seen.add(seed_id)
        question = item.get("problem") or item.get("question") or item.get("description")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing seed question/description: {seed_id}")
        declared = item.get("images") or item.get("image") or []
        declared = [declared] if isinstance(declared, str) else declared
        images = []
        for image in declared:
            candidates = [Path(image), Path(path).resolve().parent / image]
            if images_dir:
                candidates.insert(0, Path(images_dir) / Path(image).name)
            resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if resolved is None:
                raise FileNotFoundError(f"Missing seed image for {seed_id}: {image}")
            images.append(str(resolved))
        if not images and images_dir and not item.get("description"):
            for suffix in ("", "_images", "_images_0"):
                for extension in ("png", "jpg", "jpeg", "webp"):
                    candidate = Path(images_dir) / f"{seed_id}{suffix}.{extension}"
                    if candidate.is_file():
                        images = [str(candidate.resolve())]
                        break
                if images:
                    break
            if not images:
                raise FileNotFoundError(f"Missing seed image for {seed_id}")
        seeds.append({"seed_id": seed_id, "question": question.replace("<image>", "").strip(), "images": images})
        if limit and len(seeds) >= limit:
            break
    if not seeds:
        raise ValueError("No seeds selected")
    return seeds
