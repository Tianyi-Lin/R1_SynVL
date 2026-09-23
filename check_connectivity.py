import argparse
import getpass
import math
import re
import sys
import time
from pathlib import Path

from configuration import connection_value, load_config, validate_config, validate_connection
from model_api import NativeClient, completion_arguments


class ProbeError(Exception):
    pass


def probe_model(config, model, timeout, max_tokens):
    probe = {**model, "max_tokens": min(model.get("max_tokens", max_tokens), max_tokens)}
    options = {
        "api_key": connection_value(config, model, "api_key"),
        "base_url": connection_value(config, model, "base_url"),
        "timeout": timeout,
        "max_retries": 0,
    }
    if model.get("provider", "openai") == "openai":
        from openai import OpenAI

        with OpenAI(**options) as client:
            response = client.chat.completions.create(**completion_arguments(probe, "hello"))
        if not response.choices:
            raise ProbeError("No response choices")
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ProbeError("Response truncated; increase --max-tokens")
        reply = choice.message.content
    else:
        reply = NativeClient(probe, **options).generate("hello")
    if not isinstance(reply, str) or not reply.strip():
        raise ProbeError("No nonempty final text response")
    return len(reply.strip())


def http_failure(status):
    if status in (401, 403):
        detail = "check credentials and model access"
    elif status == 404:
        detail = "check endpoint path and model ID"
    elif status == 429:
        detail = "rate limit or quota exceeded"
    elif status >= 500:
        detail = "service or upstream error"
    else:
        detail = "check request parameters and protocol"
    return f"HTTP {status}; {detail}"


def failure_reason(error):
    if isinstance(error, ProbeError):
        return str(error)
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return http_failure(status)
    if isinstance(error, RuntimeError):
        status = re.fullmatch(r"(?:gemini|anthropic) request failed with HTTP ([0-9]{3})", str(error))
        if status:
            return http_failure(int(status.group(1)))
        if str(error) in ("gemini connection failed", "anthropic connection failed"):
            return "Connection failed; check network, TLS, endpoint or timeout"
    if isinstance(error, TimeoutError) or type(error).__name__ == "APITimeoutError":
        return "Request timed out"
    if type(error).__name__ == "APIConnectionError":
        return "Connection failed; check network, TLS and endpoint"
    if isinstance(error, ValueError):
        response_errors = {
            "Gemini returned no candidates": "Gemini returned no candidates; check model access and protocol",
            "Gemini output was truncated; increase max_tokens": "Gemini response truncated; increase --max-tokens",
            "Claude output was truncated; increase max_tokens": "Claude response truncated; increase --max-tokens",
            "gemini returned no final text": "Gemini returned no final text",
            "anthropic returned no final text": "Claude returned no final text",
        }
        if str(error) in response_errors:
            return response_errors[str(error)]
    if isinstance(error, (ValueError, TypeError, KeyError, IndexError, AttributeError)):
        return "Invalid or incomplete response; check protocol, model and token limit"
    return f"Request failed ({type(error).__name__}); response details omitted to protect credentials"


def check_models(config, timeout=30, max_tokens=128, check_all=False, model_names=None):
    validate_config(config)
    if not math.isfinite(timeout) or timeout <= 0 or type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("timeout and max_tokens must be positive")
    if model_names and set(model_names) - {model["name"] for model in config["models"]}:
        raise ValueError("--model must name a configured model")
    models = [model for model in config["models"] if not model_names or model["name"] in model_names]
    for model in models:
        validate_connection(config, model, model["name"])
    results = []
    for model in models:
        name = model["name"]
        print(f"[CHECK] {name}: sending hello (timeout={timeout:g}s, retries=0)", flush=True)
        started = time.monotonic()
        try:
            characters = probe_model(config, model, timeout, max_tokens)
        except Exception as error:
            reason = failure_reason(error)
            elapsed = time.monotonic() - started
            results.append({"model": name, "ok": False, "seconds": elapsed, "error": reason})
            print(f"[FAIL] {name}: {reason} ({elapsed:.2f}s)", file=sys.stderr, flush=True)
            if not check_all:
                break
        else:
            elapsed = time.monotonic() - started
            results.append({"model": name, "ok": True, "seconds": elapsed, "response_characters": characters})
            print(f"[PASS] {name}: nonempty final text, {characters} characters ({elapsed:.2f}s)", flush=True)
    passed = sum(result["ok"] for result in results)
    print(f"Connectivity: {passed}/{len(results)} checked models passed; "
          f"{len(models) - len(results)} skipped. No dataset/image/synthesis calls.", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Send hello once to each configured text model; no synthesis or image generation")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--timeout", type=float, default=30, help="Per-request timeout in seconds; default: 30")
    parser.add_argument("--max-tokens", type=int, default=128, help="Response token cap; default: 128")
    parser.add_argument("--all", dest="check_all", action="store_true", help="Report failures immediately but continue checking remaining models")
    parser.add_argument("--model", action="append", help="Check only this configured model name; can be repeated")
    parser.add_argument("--prompt-api-key", action="store_true", help="Read one hidden shared key for this check only; do not store it")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.max_tokens < 1:
        parser.error("--timeout and --max-tokens must be positive")
    if args.prompt_api_key and not sys.stdin.isatty():
        parser.error("--prompt-api-key requires an interactive terminal; otherwise use API key environment variables")
    try:
        config = load_config(args.config)
        validate_config(config)
    except Exception:
        print("[FAIL] Cannot load valid configuration; check the file path, YAML/JSON syntax and required fields.", file=sys.stderr, flush=True)
        return 2
    if args.prompt_api_key:
        key = getpass.getpass("Shared API key (hidden; not saved): ")
        if not key.strip():
            print("[FAIL] API key cannot be empty.", file=sys.stderr, flush=True)
            return 2
        for model in config["models"]:
            model["api_key"] = key
    try:
        results = check_models(config, args.timeout, args.max_tokens, args.check_all, args.model)
    except ValueError as error:
        print(f"[FAIL] Configuration: {error}", file=sys.stderr, flush=True)
        return 2
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[FAIL] Connectivity check interrupted.", file=sys.stderr, flush=True)
        sys.exit(130)
