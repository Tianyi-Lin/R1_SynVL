import os
from pathlib import Path
from urllib.parse import urlsplit

from model_api import PROVIDERS
from seed_data import DEFAULT_DATASET, validate_dataset_config
from storage import project_path, read_json, require_text


def load_config(path):
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as error:
            raise ValueError("YAML configuration requires PyYAML; run uv sync --locked in cads/") from error
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    config.setdefault("max_iterations", 10)
    config.setdefault("questions_per_seed", len(config.get("models", [])) * config["max_iterations"])
    config.setdefault("target_size", None)
    config.setdefault("seed_memory_dir", "seed_memory")
    if not isinstance(config.get("dataset", {}), dict):
        raise ValueError("dataset must be a mapping")
    config["dataset"] = {**DEFAULT_DATASET, **config.get("dataset", {})}
    config.setdefault("image_generation", {
        "provider": config.get("image_provider", "google"),
        "model": config.get("image_model", "gemini-3-pro-image"),
        "api_key_env": config.get("image_api_key_env", "CADS_IMAGE_API_KEY"),
        "base_url_env": config.get("image_base_url_env", "CADS_IMAGE_BASE_URL"),
    })
    image_config = config["image_generation"]
    if image_config.get("provider") == "openrouter" and "resolution" not in image_config and "image_size" not in image_config:
        image_config["resolution"] = "1K"
    if config["seed_memory_dir"] is not None:
        config["seed_memory_dir"] = str(project_path(config["seed_memory_dir"]))
    config["dataset"]["image_cache_dir"] = str(project_path(config["dataset"]["image_cache_dir"]))
    return config


def public_config(value):
    if isinstance(value, dict):
        return {key: "<redacted>" if key == "api_key" else public_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_config(item) for item in value]
    return value


def connection_value(config, model, field):
    source = model if field in model or f"{field}_env" in model else config
    return source.get(field) or os.environ.get(source.get(f"{field}_env", ""), "")


def validate_connection(config, model, name):
    for field in ("base_url", "api_key"):
        value = connection_value(config, model, field)
        if not isinstance(value, str) or not value.strip():
            variable = model.get(f"{field}_env", config.get(f"{field}_env", field))
            raise ValueError(f"Missing {field} for {name}; set {variable}")
    try:
        address = urlsplit(connection_value(config, model, "base_url"))
        valid = address.scheme in ("http", "https") and address.hostname and not (
            address.username or address.password or address.query or address.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"Invalid base_url for {name}; use an HTTP(S) API base without credentials, query or fragment")


def validate_config(config, live=False):
    validate_dataset_config(config.get("dataset", DEFAULT_DATASET))
    models = config["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a nonempty list")
    for model in models:
        require_text(model, ("name", "model", "visual_mode"))
        if model.get("provider", "openai") not in PROVIDERS:
            raise ValueError(f"Unknown API provider for {model['name']}")
        if model.get("provider") == "anthropic" and (type(model.get("max_tokens")) is not int or model["max_tokens"] < 1):
            raise ValueError("Anthropic models require a positive integer max_tokens")
    names = [model["name"] for model in models]
    if len(set(names)) != len(names):
        raise ValueError("Collective model names must be unique")
    if len({model["model"] for model in models}) != len(models):
        raise ValueError("Repeated model endpoints do not count as independent judges")
    if type(config["max_iterations"]) is not int or not 1 <= config["max_iterations"] <= 10:
        raise ValueError("max_iterations must be an integer from 1 to 10")
    if type(config["questions_per_seed"]) is not int or config["questions_per_seed"] < 1:
        raise ValueError("questions_per_seed must be a positive integer")
    if config["target_size"] is not None and (type(config["target_size"]) is not int or config["target_size"] < 1):
        raise ValueError("target_size must be null or a positive integer")
    if type(config.get("json_attempts", 3)) is not int or config.get("json_attempts", 3) < 1:
        raise ValueError("json_attempts must be positive")
    memory_directory = config.get("seed_memory_dir", "seed_memory")
    if memory_directory is not None and (not isinstance(memory_directory, str) or not memory_directory.strip()):
        raise ValueError("seed_memory_dir must be a nonempty path string or null")
    for role in ("reflection_model", "equivalence_model"):
        if config[role] not in names:
            raise ValueError(f"Unknown model for {role}")
    for model in models:
        if model["visual_mode"] not in ("native", "text_only", "caption", "unresolved"):
            raise ValueError("visual_mode must be native, text_only, caption or unresolved")
        if live and model["visual_mode"] == "unresolved":
            raise ValueError(f"Resolve visual input for {model['name']} before running; see README.md")
    if any(model["visual_mode"] == "caption" for model in models):
        caption_model = next((model for model in models if model["name"] == config.get("caption_model")), None)
        if caption_model is None or caption_model["visual_mode"] != "native":
            raise ValueError("caption_model must name a model accepting native image input")
    image_config = config["image_generation"]
    if type(image_config.get("enabled", True)) is not bool:
        raise ValueError("image_generation.enabled must be boolean")
    if image_config.get("provider", "google") not in ("google", "openrouter"):
        raise ValueError("image_generation.provider must be google or openrouter")
    if image_config.get("enabled", True):
        require_text(image_config, ("model",))
    for field in ("api_version", "aspect_ratio", "resolution"):
        if field in image_config:
            require_text(image_config, (field,))
    if "resolution" in image_config and "image_size" in image_config:
        raise ValueError("Use resolution for OpenRouter or image_size for Google, not both")
    image_size = image_config.get("resolution", image_config.get("image_size", "1K"))
    if image_size not in ("1K", "2K", "4K"):
        raise ValueError("Nano Banana Pro resolution must be 1K, 2K or 4K")
    if live:
        if not image_config.get("enabled", True):
            raise ValueError("Image generation is disabled; set image_generation.enabled to true only when ready to run")
        for model in models:
            validate_connection(config, model, model["name"])
        validate_connection({}, image_config, "image_generation")
