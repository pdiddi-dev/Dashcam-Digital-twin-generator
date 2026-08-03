"""Shared NVIDIA hosted-API client + video encoding helpers."""

import base64
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
MAX_INLINE_BYTES = 25 * 1024 * 1024


def encode_video(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if not mime or not mime.startswith("video/"):
        mime = "video/mp4"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def check_size(path: Path) -> None:
    size = path.stat().st_size
    if size > MAX_INLINE_BYTES:
        mb = size / 1024 / 1024
        raise ValueError(
            f"{path.name} is {mb:.1f} MB, over the "
            f"{MAX_INLINE_BYTES // 1024 // 1024} MB inline limit"
        )


def make_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set (see .env.example)")
    return OpenAI(base_url=BASE_URL, api_key=api_key)
