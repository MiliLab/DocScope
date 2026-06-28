"""Common Backend descriptor used by `infer/run_infer.py`.

Every backend (Claude / Gemini / OpenAI-compatible) exposes one of these.
The unified runner only needs to know:
  - Which env var holds the API key.
  - Whether to feed images as raw base64 strings or as `data:` URIs.
  - A single `call(question, image_inputs) -> (text, thinking, usage)` callable.

`thinking` is the model's extended-reasoning trace if the provider exposes one
(Anthropic thinking blocks / OpenAI `reasoning_content` / Gemini thoughts);
empty string when unavailable. Backends MUST `try/except` around any thinking
extraction so a missing field never breaks inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

ImageFormat = Literal["base64", "data_uri"]
# page_numbers: optional list of original 1-based page numbers (one per image);
# backends use them as the GLOBAL PAGE marker values instead of 1, 2, 3...
CallFn = Callable[[str, list[str], "list[int] | None"], "tuple[str, str, dict]"]


@dataclass
class Backend:
    name: str
    api_key_env: str
    image_format: ImageFormat
    model: str
    call: CallFn
    simple_prompt: bool = False
