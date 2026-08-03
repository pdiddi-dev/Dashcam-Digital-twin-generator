"""Stage 1: call the NVIDIA VLM, get a structured Scene back."""

import json
import re
from pathlib import Path
from typing import Literal

from openai import APIError
from pydantic import BaseModel, Field, ValidationError, field_validator

from nvidia_client import MODEL, check_size, encode_video, make_client


class Environment(BaseModel):
    road_type: Literal["residential", "highway", "urban", "rural", "parking"]
    lanes: int = Field(ge=1, le=8)
    weather: Literal["clear", "overcast", "rain", "snow", "fog"]
    time_of_day: Literal["day", "dusk", "night", "dawn"]
    sun_azimuth_deg: int = Field(ge=0, le=360, default=180)


class Ego(BaseModel):
    speed_qualitative: Literal["stationary", "slow", "moderate", "fast"]
    heading: Literal["straight", "left_turn", "right_turn"]


class Actor(BaseModel):
    type: Literal["car", "truck", "van", "bus", "pedestrian", "cyclist", "motorcycle"]
    lane: Literal[
        "same_ahead", "same_behind",
        "oncoming_left", "oncoming_ahead",
        "left_shoulder", "right_shoulder",
        "parked_left", "parked_right",
        "crosswalk",
    ]
    distance_qualitative: Literal["very_close", "close", "medium", "far"]
    color: str = "gray"
    notable: str = ""


class EnvironmentFeature(BaseModel):
    type: Literal["trees", "buildings", "barrier", "signs", "bollards", "streetlights"]
    side: Literal["left", "right", "both"]


class Scene(BaseModel):
    environment: Environment
    ego: Ego
    actors: list[Actor]
    environment_features: list[EnvironmentFeature] = []
    summary: str

    @field_validator("environment_features", mode="before")
    @classmethod
    def _drop_invalid_features(cls, v):
        if not isinstance(v, list):
            return []
        out = []
        for item in v:
            try:
                out.append(EnvironmentFeature.model_validate(item))
            except ValidationError:
                continue
        return out


EXTRACTION_PROMPT = """You are a computer vision system extracting a structured scene from dashcam video.

CRITICAL OUTPUT RULES:
- Return ONLY the JSON object below. Nothing before it, nothing after it.
- Do NOT think aloud. Do NOT enumerate frames. Do NOT explain your reasoning.
- Do NOT wrap the output in markdown fences.
- The very first character of your response MUST be `{` and the very last MUST be `}`.

Schema:

{
  "environment": {
    "road_type": "residential" | "highway" | "urban" | "rural" | "parking",
    "lanes": <int 1-8>,
    "weather": "clear" | "overcast" | "rain" | "snow" | "fog",
    "time_of_day": "day" | "dusk" | "night" | "dawn",
    "sun_azimuth_deg": <0-360, direction sunlight comes from; 180 = behind camera, 0 = ahead>
  },
  "ego": {
    "speed_qualitative": "stationary" | "slow" | "moderate" | "fast",
    "heading": "straight" | "left_turn" | "right_turn"
  },
  "actors": [
    {
      "type": "car" | "truck" | "van" | "bus" | "pedestrian" | "cyclist" | "motorcycle",
      "lane": "same_ahead" | "same_behind" | "oncoming_left" | "oncoming_ahead" | "left_shoulder" | "right_shoulder" | "parked_left" | "parked_right" | "crosswalk",
      "distance_qualitative": "very_close" | "close" | "medium" | "far",
      "color": "<lowercase color word, e.g. red, black, silver>",
      "notable": "<optional one-clause note, else empty string>"
    }
  ],
  "environment_features": [
    {"type": "trees" | "buildings" | "barrier" | "signs" | "bollards" | "streetlights", "side": "left" | "right" | "both"}
  ],
  "summary": "<one sentence describing the scene>"
}

Rules:
- Do NOT invent absolute distances or speeds. Use only the qualitative buckets above.
- List every distinct actor you can see, including parked vehicles (as "parked_left" or "parked_right").
- If uncertain about a value, pick the most plausible option — do not omit fields.
- Output MUST parse as JSON. No comments, no trailing commas, no markdown fences."""


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` markdown fences if the model wraps its output."""
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1) if m else text


def _extract_json_object(text: str) -> str:
    """Grab the first balanced {...} block. Handles prose before/after."""
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def extract_scene(video: Path) -> Scene:
    check_size(video)
    client = make_client()
    video_url = encode_video(video)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXTRACTION_PROMPT},
                        {"type": "video_url", "video_url": {"url": video_url}},
                    ],
                }
            ],
            max_tokens=6144,
        )
    except APIError as e:
        raise RuntimeError(f"NVIDIA API {e.status_code}: {e.message}") from e

    raw = response.choices[0].message.content or ""
    candidate = _extract_json_object(_strip_fences(raw))

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"model returned non-JSON output:\n---\n{raw}\n---\n(parse error: {e})"
        ) from e

    try:
        return Scene.model_validate(data)
    except ValidationError as e:
        raise RuntimeError(
            f"model output failed schema validation:\n---\n{json.dumps(data, indent=2)}\n---\n{e}"
        ) from e
