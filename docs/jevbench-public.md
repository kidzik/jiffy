# JevBench Public-Subset Evaluation

Measured 2026-09-21 on one H100 80GB, frozen BF16 DiffusionGemma, one decoder
step, default shared-document JevProtocol. No benchmark-specific training,
prompt tuning, retries, or answer selection. One request per task, serially.

Upstream: [JevBench](https://github.com/fstandhartinger/jevbench), commit
`6f4a36c9b73c8b152baca383bce5cf2f59ae9f38`, results revision v1.2.13.
The runner uses upstream `TypeSafeAdapter.build_request`, `score_task`, and
metric functions. Only state and question enter inference; expected answers,
rationales and gold distributions are reserved for scoring.

## Accuracy on Identical Public Items

| System | Easy (48) | Standard (72) | Hard (111) | Total (231) |
| --- | ---: | ---: | ---: | ---: |
| Jiffy, this run | 48 (100%) | 68 (94.4%) | 79 (71.2%) | 195 (84.4%) |
| Jev 1.13.0, published | 48 (100%) | 71 (98.6%) | 81 (73.0%) | 200 (86.6%) |
| djev, published | 48 (100%) | 71 (98.6%) | 75 (67.6%) | 194 (84.0%) |
| OpenJev DiffusionGemma NVFP4, published | 48 (100%) | 70 (97.2%) | 71 (64.0%) | 189 (81.8%) |
| openjev-sglang, published | 48 (100%) | 68 (94.4%) | 81 (73.0%) | 197 (85.3%) |
| GPT-5.6 Luna low, published | 48 (100%) | 70 (97.2%) | 107 (96.4%) | 225 (97.4%) |

Comparison counts are recomputed from upstream public per-task outcomes on
the same 231 IDs, not copied from full-suite headline scores. Other systems
were not re-run here. A difference of one or five decisions is not evidence
of a statistically established ranking.

All 231 answers were valid. Raw local latency was **259 ms median / 588 ms p95**
over the whole public subset, with model loading and a separate non-benchmark
warm-up excluded. This is not end-to-end latency from the benchmark's Germany
client, nor performance under concurrent load.

## Weaknesses

- Temporal/numeric reasoning: 6/15 (40%).
- Probability tasks: 6/10 (60%).
- Long policy tasks: 13/19 (68.4%).
- Multi-hop tasks: 13/18 (72.2%).
- Hard-tier top-label ECE: 0.2036, indicating substantial calibration error.
- Mean total-variation distance to exact gold distributions: 0.4302 on 10 items.

Public-subset Intelligence is **85.83**, using the upstream tier weights with
the absent judge tier removed and the remaining weights renormalized.
Public-hard Calibration is **58.13**. These are partial-data metrics, not
official leaderboard scores or estimates of unseen-item performance.

## Why There Is No Official Composite

The linked `w=33-0-33-33` view geometrically combines Intelligence, Speed and
Cost equally, excluding Calibration. The complete suite has 534 decisions:
231 are published here, 146 imported judge tasks are not redistributed, and
157 easy/standard/hard decisions are private. We cannot reproduce all 534
from this public repository.

The official Speed cohort is standard + judge, not our public-only cohort.
Our standard-public-only proxy is 81.20 after the site's self-hosted adjustment
of 2x latency + 0.15 s, but it is not an official Speed score. There is no
Jiffy hosting tariff, so Cost and the requested composite are deliberately
left null, not treated as zero-dollar hosting or perfect Cost. An official
full score requires the missing data (or an independent maintainer-run
evaluation) and a disclosed, defensible cost basis.

Public tasks are not a held-out contamination-proof test. This run did not tune
on them, but the base model's exposure to public benchmark material is unknown.

## Reproduce

```bash
gh repo clone https://github.com/fstandhartinger/jevbench.git /tmp/jevbench
git -C /tmp/jevbench checkout --detach 6f4a36c9b73c8b152baca383bce5cf2f59ae9f38
python -m benchmarks.jevbench --upstream /tmp/jevbench --output artifacts/jevbench-new-run
```

Install Jiffy's pinned runtime first. This command requires the GPU and model
checkpoint; upstream tasks and harness remain separately licensed MIT assets.
The output directory must not already exist. It records task and source hashes,
every prediction, latency, validity, error, and aggregate comparisons.

This run's local artifacts: `artifacts/jevbench-public-20260921/manifest.json`,
`predictions.jsonl`, and `report.json`. Artifacts are ignored by git.
The same files are published as `jevbench-public-20260921.zip` with the
[v0.1.0 release](https://github.com/kidzik/jiffy/releases/tag/v0.1.0).
