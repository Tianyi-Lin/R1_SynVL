import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


PROVIDERS = ("openai", "gemini", "anthropic")


def completion_arguments(model, content):
    arguments = {"model": model["model"], "messages": [{"role": "user", "content": content}]}
    for field in ("temperature", "max_tokens"):
        if field in model:
            arguments[field] = model[field]
    return arguments


def native_parts(content, provider):
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
    parts = []
    for block in blocks:
        if block["type"] == "text":
            parts.append({"text": block["text"]} if provider == "gemini" else dict(block))
        elif block["type"] == "image_url":
            header, separator, encoded = block["image_url"]["url"].partition(",")
            if not separator or not header.startswith("data:image/") or not header.endswith(";base64") or not encoded:
                raise ValueError("Native adapters require a base64 image data URL")
            mime = header[len("data:"):].split(";", 1)[0]
            if provider == "gemini":
                parts.append({"inlineData": {"mimeType": mime, "data": encoded}})
            else:
                parts.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": encoded}})
        else:
            raise ValueError(f"Unsupported input block: {block['type']}")
    return parts


def native_request(model, base_url, api_key, content):
    provider = model["provider"]
    headers = {"Content-Type": "application/json"}
    if provider == "gemini":
        model_id = quote(model["model"].removeprefix("models/"), safe="")
        url = base_url.rstrip("/") + f"/models/{model_id}:generateContent?" + urlencode({"key": api_key})
        body = {"contents": [{"role": "user", "parts": native_parts(content, provider)}]}
        generation = {}
        if "temperature" in model:
            generation["temperature"] = model["temperature"]
        if "max_tokens" in model:
            generation["maxOutputTokens"] = model["max_tokens"]
        if generation:
            body["generationConfig"] = generation
    elif provider == "anthropic":
        url = base_url.rstrip("/") + "/messages"
        headers.update({"x-api-key": api_key, "anthropic-version": model.get("anthropic_version", "2023-06-01")})
        body = completion_arguments(model, content if isinstance(content, str) else native_parts(content, provider))
        if "max_tokens" not in body:
            raise ValueError("Anthropic Messages requires max_tokens")
    else:
        raise ValueError(f"Unsupported native provider: {provider}")
    return url, headers, body


def native_response_text(provider, response):
    if provider == "gemini":
        candidates = response.get("candidates") or []
        if not candidates:
            raise ValueError("Gemini returned no candidates")
        candidate = candidates[0]
        if candidate.get("finishReason") == "MAX_TOKENS":
            raise ValueError("Gemini output was truncated; increase max_tokens")
        blocks = candidate.get("content", {}).get("parts", [])
        text = "".join(block["text"] for block in blocks if isinstance(block.get("text"), str) and not block.get("thought"))
    elif provider == "anthropic":
        if response.get("stop_reason") == "max_tokens":
            raise ValueError("Claude output was truncated; increase max_tokens")
        text = "".join(block["text"] for block in response.get("content", []) if block.get("type") == "text" and isinstance(block.get("text"), str))
    else:
        raise ValueError(f"Unsupported native provider: {provider}")
    if not text.strip():
        raise ValueError(f"{provider} returned no final text")
    return text


class NativeClient:
    def __init__(self, model, base_url, api_key, timeout=180, max_retries=2):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(self, content):
        url, headers, body = native_request(self.model, self.base_url, self.api_key, content)
        request = Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST")
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return native_response_text(self.model["provider"], data)
            except HTTPError as error:
                status = error.code
                error.close()
                if status not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    raise RuntimeError(f"{self.model['provider']} request failed with HTTP {status}") from None
            except (URLError, TimeoutError):
                if attempt == self.max_retries:
                    raise RuntimeError(f"{self.model['provider']} connection failed") from None
            time.sleep(2 ** attempt)
