#!/usr/bin/env python3
"""Semantic search over a fleet of extracted dashcam scenes.

Reads every `scene.json` produced by `build_twin.py`, embeds a compact
description with NeMo Retriever (`nvidia/nemotron-3-embed-1b`), and answers
natural-language queries via cosine similarity.

Usage:
    python fleet_search.py index [--twins-dir twins/]           # build/refresh
    python fleet_search.py query "oncoming vehicles in rain"    # search
    python fleet_search.py stats                                # index summary
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from nvidia_client import EMBED_MODEL, make_client
from scene_extractor import Scene

INDEX_PATH = Path("fleet_index.npz")


def _describe_scene(scene: Scene, source: str) -> str:
    """Compact natural-language description of a Scene — one line per attribute.
    Concatenated with the model-generated summary. Feeds the embedder."""
    env = scene.environment
    parts = [
        f"Source: {source}.",
        scene.summary,
        f"Road: {env.road_type}, {env.lanes}-lane, {env.weather}, {env.time_of_day}.",
        f"Ego: {scene.ego.speed_qualitative} speed, {scene.ego.heading}.",
    ]
    if scene.actors:
        actor_bits = [
            f"{a.color} {a.type} ({a.lane}, {a.distance_qualitative})"
            for a in scene.actors
        ]
        parts.append("Actors: " + "; ".join(actor_bits) + ".")
    if scene.environment_features:
        parts.append(
            "Features: "
            + ", ".join(f"{f.type} on {f.side}" for f in scene.environment_features)
            + "."
        )
    return " ".join(parts)


def _embed(client, texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)


def cmd_index(args) -> int:
    scene_files = sorted(Path(args.twins_dir).glob("*/scene.json"))
    if not scene_files:
        print(f"error: no scene.json files under {args.twins_dir}/", file=sys.stderr)
        print("hint: run `python build_twin.py <clip.mp4>` on a few clips first.", file=sys.stderr)
        return 2

    descriptions = []
    sources = []
    for f in scene_files:
        scene = Scene.model_validate(json.loads(f.read_text()))
        sources.append(f.parent.name)
        descriptions.append(_describe_scene(scene, f.parent.name))

    print(f"embedding {len(descriptions)} scenes via {EMBED_MODEL}...", file=sys.stderr)
    client = make_client()
    embeddings = _embed(client, descriptions)

    np.savez(
        INDEX_PATH,
        embeddings=embeddings,
        sources=np.array(sources),
        descriptions=np.array(descriptions),
    )
    print(f"wrote {INDEX_PATH} — {len(sources)} scenes, {embeddings.shape[1]}-d vectors")
    return 0


def cmd_query(args) -> int:
    if not INDEX_PATH.exists():
        print(f"error: {INDEX_PATH} not found. Run `fleet_search.py index` first.", file=sys.stderr)
        return 2

    data = np.load(INDEX_PATH, allow_pickle=True)
    embeddings = data["embeddings"]
    sources = data["sources"]
    descriptions = data["descriptions"]

    client = make_client()
    q_vec = _embed(client, [args.query])[0]

    scores = embeddings @ q_vec
    order = np.argsort(-scores)[: args.top_k]

    print(f"\nQuery: {args.query!r}")
    print(f"Corpus: {len(sources)} scene(s)\n")
    for rank, idx in enumerate(order, 1):
        print(f"[{rank}] score={scores[idx]:.3f}  source={sources[idx]}")
        print(f"    {descriptions[idx]}\n")
    return 0


def cmd_stats(args) -> int:
    if not INDEX_PATH.exists():
        print(f"no index at {INDEX_PATH}. Run `fleet_search.py index` first.")
        return 0
    data = np.load(INDEX_PATH, allow_pickle=True)
    n, dim = data["embeddings"].shape
    print(f"index: {INDEX_PATH}")
    print(f"scenes: {n}")
    print(f"embedding dim: {dim}")
    print(f"model: {EMBED_MODEL}")
    print(f"sources: {', '.join(data['sources'].tolist())}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Build or refresh the fleet embedding index")
    p_index.add_argument("--twins-dir", default="twins", help="Directory containing per-clip subdirs with scene.json")
    p_index.set_defaults(func=cmd_index)

    p_query = sub.add_parser("query", help="Search the indexed fleet")
    p_query.add_argument("query", help="Natural-language query, e.g. 'oncoming cars at night'")
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.set_defaults(func=cmd_query)

    p_stats = sub.add_parser("stats", help="Print index summary")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
