# Evaluation Notes

These measurements were obtained before extraction into this standalone package.
The decoder/scoring implementation and pinned model revision are unchanged;
the new package has its own isolated smoke check. Full historical artifacts are
retained in the private `kidzik/jiffy-legacy` repository's associated workspace,
not bundled here and not required for inference or tests.

## Image and Long-Context Pilot

Model: `google/diffusiongemma-26B-A4B-it`, revision
`f7f5b7f5fa82ffc52addd066915886d497f5517b`. H100 80GB, BF16, SDPA,
grouped-matrix-multiply experts; no quantization or task-specific training.

| Cohort | Decisions correct | Independent images |
| --- | --- | --- |
| Public images | 19/22 | 11 |
| Short procedural image-ledger joins | 18/18 | 6 |
| Near-32K procedural joins | 9/9 | 3 |

The public cohort used ScienceQA, AOKVQA, TAT-QA and InfographicVQA; pretraining
contamination is possible. Derived verification questions are not independent
documents. The procedural tasks do not establish natural long-document accuracy.

Selected-position vocabulary readout matched the full-canvas one-step reference
on all 49 decisions, with zero probability total-variation distance. Short-request
median was 251 ms after image warmup. Near-32K median was about 6.64 seconds,
dominated by encoder prefill. Measurements exclude model loading, HTTP and
contract compilation. Actual image tokens were approximately 530 for short
requests; the configured vision budget was 560. Peak allocated GPU memory in
the long-context pilot was 66.39 GiB; allocator reservation was larger.

Small probes checked cache immutability, question/option ordering and shared
versus separate scoring. These are implementation checks, not universal
invariance or calibration guarantees.

## Doom

The same four typed questions and shared controller ran Freedoom2 map01, skill 3,
seeds 20260920-20260922, eight tics/action and a 100-game-second episode limit.
Inputs were privileged engine/map state, not screenshots. All initial state
hashes matched; live trajectories subsequently diverged.

| Policy | Kills across three seeds | Level exits | Median on 96 identical states |
| --- | --- | --- | --- |
| Jev | 20 | 0 | 173 ms |
| DiffusionGemma | 15 | 0 | 174 ms |

Jev timings were historical and network-inclusive. DiffusionGemma ran locally.
This is not a simultaneous kernel-speed comparison or evidence of equivalent
general intelligence. All 1279 live decisions and 96 replay decisions returned
validated structures. Navigation remained unreliable and all exit rewards were zero.

## Probability Semantics

Scores are softmax-normalized candidate logits, untempered and subject to the
model's native logit softcapping. The fixed answer scaffold and denoising state
condition the scores. Bidirectional decoding can couple answers; these are not
independent joint marginals. Do not equate high confidence with verified evidence.
