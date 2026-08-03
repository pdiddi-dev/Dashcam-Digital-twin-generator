# Case study — from dashcam to digital twin

## The problem

Fleet operators, robotaxi programs, and OEM safety teams collectively record **billions of hours** of dashcam footage every year. Almost none of it is directly usable by the tools that would benefit from it most: **simulation and scenario-testing platforms**. Video is a lossy, unstructured medium; a scenario engine expects a scene graph — actors, positions, lanes, weather, ego state — in a schema it can consume.

The consequence: safety engineers rediscover the same corner cases over and over from telemetry alone, and iSVs building AV testing tools ship without a natural on-ramp for a partner's existing video assets.

## Who feels this pain

- **iSVs in the AV testing space** (Foretellix, Applied Intuition, IPG CarMaker, and adjacent): every enterprise conversation eventually hits *"can you ingest our fleet video?"* — and the honest answer is usually "not directly, you have to hand-author scenarios."
- **OEM safety and validation teams**: sit on internal dashcam corpora that can't be replayed in their simulator without weeks of manual scene reconstruction per scenario.
- **Fleet operators and insurers**: have compelling video evidence they can't turn into structured incident reports or synthetic training data.

## What the NVIDIA stack enables

The primitives to close this gap already exist in NVIDIA's stack — they've just never been chained together in a small enough package for a partner to fork:

- **Cosmos / Cosmos-Reason (physical AI VLMs)** — reason about video scenes, actors, physics, and behavior. Available today via hosted API on build.nvidia.com and as downloadable NIM containers.
- **OpenUSD + Omniverse** — the interchange format and scene composition runtime that every serious AV simulator either speaks natively or imports.
- **NVIDIA NIM inference services** — production-ready hosted endpoints so partners don't need to stand up their own GPU inference to try the pipeline.

The insight: **treat Cosmos as the ingest bridge from unstructured video into a structured USD stage that the rest of the ecosystem already knows how to consume.**

## This pipeline

Two stages, ~600 lines of Python, one command:

```
dashcam.mp4  ─►  [ NVIDIA hosted VLM ]  ─►  scene.json  ─►  [ USD builder ]  ─►  scene.usda
                 (schema-constrained)      (contract)      (deterministic)      (Omniverse-ready)
```

**Stage 1 — extraction.** Base64-inline the clip, one hosted-API call to a Cosmos-family reasoning model, JSON-schema-constrained prompt, Pydantic validation. The output is a small, diff-able, schema-versionable scene contract — not a black-box embedding.

**Stage 2 — composition.** Deterministic Python that maps the qualitative scene contract onto an OpenUSD stage (ego, actors, road geometry, sidewalks, curbs, trees, sun angle, sky). Human-readable ASCII `.usda`, opens in Omniverse USD Composer, Reality Composer Pro, Blender, or `usdview`.

Zero training. No GPU required on the client. The entire dev loop runs from a laptop.

## What a partner does with it

A concrete first-week onboarding for an iSV partner adopting this pipeline:

1. **Day 1** — clone the repo, replace the primitive vehicle geometry with references to the partner's existing USD asset library. Repointing takes an afternoon.
2. **Day 2-3** — bolt the output into the partner's scenario runner. Because the output is OpenUSD, this is usually a `usdImport` call, not a custom parser.
3. **Day 4-5** — extend the extraction schema for domain-specific fields the partner cares about (traffic-light state, ego signals, road-surface condition). Prompt engineering, not model training.
4. **Week 2+** — layer Cosmos Predict on top to generate scene variations (night, rain, added pedestrian) for corner-case augmentation. Each variation becomes a new scenario the partner's simulator can now cover.

## Metrics that would matter to a partner-success motion

- **Time-to-first-scenario** for a new partner: baseline is weeks of manual authoring. Target with this pipeline: <1 hour from an unseen video clip to a scenario replaying in the partner's simulator.
- **Scenario coverage growth per week** for an existing partner: how many net-new scenarios does the pipeline generate from their existing video corpus that a human wouldn't have written from scratch.
- **Corner-case hit rate**: of scenes flagged by Cosmos-Reason as anomalous, how many actually reproduce a real edge case in the partner's test suite.
- **Adoption depth**: fraction of partner's active scenarios sourced from video ingestion vs. hand-authored six months in.

None of these need the partner to instrument anything new — the video already exists, and the simulator already tracks scenarios.

## If I were shaping this as a DevRel program

An honest 90-day rollout:

- **Week 0-2 — assets.** Ship the reference implementation (this repo), a two-hour hands-on codelab, and a 15-minute demo video. Move from "here's a demo" to "here's a starter kit."
- **Week 3-6 — 3 lighthouse partners.** Recruit 3 friendly iSVs in the AV testing space for a co-development sprint. Weekly office hours. Feed their integration friction back into the reference implementation.
- **Week 7-10 — public launch.** Blog post co-authored with the lighthouse partners (concrete numbers, not marketing). Talk submission for GTC. Publish an "ingest recipes" gallery showing the pipeline extended for different domains (highway, urban, parking, off-road).
- **Week 11-13 — community flywheel.** Discord/Slack channel with the ingest recipes as the anchor. First hackathon prompt: *"extend the ingest schema for your simulator, submit a PR."*

## What this repo isn't

- Not a production system — it's a shape.
- Not photoreal — it's a scene contract with placeholder geometry, on purpose.
- Not metric-accurate — monocular VLMs can't reliably estimate distance, so the schema is qualitative on purpose.
- Not tied to a specific partner's simulator — OpenUSD is the neutral surface, and staying neutral is what makes it a partner enablement artifact rather than a bespoke integration.

The value is that it makes the *shape* of the opportunity legible in ~600 lines of Python. Every iSV that reads this repo can see exactly where their assets, their simulator, and their scenarios plug in.
