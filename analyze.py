#!/usr/bin/env python3
"""Send a short dashcam clip to NVIDIA's hosted VLM and print the analysis."""

import argparse
import sys
from pathlib import Path

from openai import APIError

from nvidia_client import MODEL, check_size, encode_video, make_client

DEFAULT_PROMPT = (
    "You are analyzing footage from a car's dashcam. Describe what is happening "
    "in the scene: the road and environment, weather and lighting, other vehicles "
    "and their behavior, and anything notable the driver should be aware of. "
    "Be concrete and specific."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to the dashcam clip")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Override the default prompt")
    parser.add_argument("--save", action="store_true", help="Write output to <video>.analysis.txt")
    args = parser.parse_args()

    if not args.video.is_file():
        print(f"error: {args.video} is not a file", file=sys.stderr)
        return 2

    try:
        check_size(args.video)
        client = make_client()
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    video_url = encode_video(args.video)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.prompt},
                        {"type": "video_url", "video_url": {"url": video_url}},
                    ],
                }
            ],
            max_tokens=1024,
        )
    except APIError as e:
        print(f"error: NVIDIA API returned {e.status_code}: {e.message}", file=sys.stderr)
        return 1

    text = response.choices[0].message.content or ""
    print(text)

    if args.save:
        out = args.video.with_suffix(args.video.suffix + ".analysis.txt")
        out.write_text(text)
        print(f"\n(saved to {out})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
