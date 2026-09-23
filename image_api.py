import base64
import binascii
import json
import time
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OpenRouterImageClient:
    def __init__(self, base_url, api_key, timeout=180, max_retries=2):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(self, model, prompt, aspect_ratio=None, resolution=None):
        body = {"model": model, "prompt": prompt, "n": 1}
        if aspect_ratio:
            body["aspect_ratio"] = aspect_ratio
        if resolution:
            body["resolution"] = resolution
        request = Request(
            self.base_url.rstrip("/") + "/images",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        response_data = None
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    response_data = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as error:
                status = error.code
                error.close()
                if status not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    raise RuntimeError(f"OpenRouter image request failed with HTTP {status}") from None
            except (URLError, TimeoutError, ConnectionError, HTTPException):
                if attempt == self.max_retries:
                    raise RuntimeError("OpenRouter image connection failed") from None
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ValueError("OpenRouter image generator returned invalid JSON") from None
            time.sleep(2 ** attempt)
        images = response_data.get("data", []) if isinstance(response_data, dict) else []
        image = images[0] if images and isinstance(images[0], dict) else {}
        encoded = image.get("b64_json")
        media_type = image.get("media_type", "image/png")
        if not isinstance(encoded, str) or not encoded or not isinstance(media_type, str) or not media_type.startswith("image/"):
            raise ValueError("OpenRouter image generator returned no image; rerun to retry")
        try:
            return base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("OpenRouter image generator returned invalid base64 image data") from None
