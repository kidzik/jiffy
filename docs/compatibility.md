# TypeSafe Protocol Compatibility

Target: the public [HTTP API](https://docs.typesafe.ai/api) and
[Score semantics](https://docs.typesafe.ai/primitives/score), checked 2026-09-21.
This is an independent Gemma implementation, not Jev or a certified replacement.
Do not describe it as 100% behaviorally equivalent.

## Implemented Surface

| Contract | Jiffy adapter |
| --- | --- |
| Endpoint | `POST /v1/systemone`, bearer authentication |
| Discovery | Authenticated `GET /v1/models`, Jiffy's own model metadata |
| Request | Required `model`, `state`, `questions` |
| State and instructions | String, JSON object, or JSON array |
| Noul | Optional structured true/false criteria; answer `type`, `noul` |
| Choice | 2-255 named options; string/object/array/null criteria |
| Score | 2-10 ordered string/object/array criteria |
| Choice answer | Selected key, complete normalized probabilities, confidence |
| Score answer | Weighted zero-based index, complete probabilities, original legend, confidence |
| Response | Actual model identity, named answers, integer token usage |
| Question identity | Caller names are not passed to the model |
| Question isolation | Shared causal document prefill; independent question cache branches |
| Score-level isolation | Each level evaluated without its index or neighbors |
| Errors | 401 authentication, 422 validation, 429 rate limit, 529 busy/OOM |

Use `JevProtocol.evaluate` for this contract in Python, or install `.[serve]`
and run `jiffy-serve`. The original `DiffusionDecisions.system_one` remains a
separate, faster experimental shared-pass interface with extra diagnostics.
The adapter does not download URLs or read paths embedded in a request.

## Explicit Differences

- Predictions and calibration are Gemma's, not Jev's. Distributions are
  uncalibrated model scores. Confidence uses one minus normalized entropy;
  TypeSafe's exact formula is not public.
- Score evaluates each rubric as a binary match, then normalizes the yes
  probabilities across levels. All-zero matches fall back to uniform. This
  preserves isolation, but is our scoring algorithm, not Jev's internal formula.
- Structured score legends preserve the original JSON values, following the
  structured example in the Score guide (the API reference lists string values).
- Default execution encodes the document (and SDK images) once. Independent
  cache branches process question suffixes and answer canvases. Branches with
  exactly equal token lengths run together, up to four per batch; different
  lengths run separately. Each branch keeps the original causal encoder and
  bidirectional diffusion decoder attention. No question can attend to another.
  This does not reproduce Jev's near-constant multi-question latency. One HTTP
  response is not one GPU pass. BF16 execution can produce numerical
  differences, including material shifts near uncertain decisions; independence
  does not imply identical probabilities.
- `usage.input_tokens` counts the shared document once plus each question
  suffix. The optional `execution="sequential"` reference instead counts
  repeated state tokens. `output_tokens` counts scored answer slots, including
  each Score level, not generated prose or padded canvas positions. It is not
  a reproduction of TypeSafe billing accounting.
- Model aliases accepted: `jiffy-diffusiongemma`, `jev-latest`, and the pinned
  Google model ID. The response never impersonates a TypeSafe model.
- Defaults: 256 questions, 2 MB request body, 60 authenticated requests/minute,
  32,768 positions per inference including the 256-token canvas. The first three
  limits are configurable in `create_app`. Invalid requests count toward rate
  limits. Overlap is rejected with 529; no queue or multi-GPU scheduling.
- Unknown fields, duplicate JSON keys and nonfinite JSON numbers are rejected.
  Unexpected inference runtime failures return 500, not a fabricated answer.
- Images remain an SDK extension, not part of the documented TypeSafe HTTP
  contract. HTTP JSON keys named `images` are ordinary evidence, not attachments.
- Choices above 26 use additional verified single-token labels. Protocol and
  tokenization support do not establish accuracy on large candidate sets.

## Release Gates

Fixture tests check serialization, validation, isolation, option limits,
authentication, rate limiting, busy/OOM errors, and lock recovery. The official
TypeSafe Python SDK 0.7.1 passes request serialization and response validation
against the ASGI endpoint for all three types and model discovery. This is not
a differential test against the hosted Jev service. Before claiming full
interoperability, run approved live conformance cases against both providers. Production serving
also needs TLS termination, deployment-specific access controls, cancellation,
load tests, and observability without logging private request bodies.

Earlier sequential H100 smoke on 2026-09-21: structured three-type request completed in 1.73 s
(four isolated evaluations); renaming/removing unrelated questions preserved
the Noul answer exactly. A 255-option inference returned a valid distribution
and correctly selected the stated number 42. These are small correctness
smokes, not accuracy or latency benchmarks.

## Shared Document Implementation

The pinned Gemma encoder is causal for text and bidirectional within vision
blocks. Its document KV cache therefore does not depend on future questions.
Each branch gets independent cache/layer metadata and initially shares read-only
document tensors; dynamic cache updates allocate new concatenated tensors.
This avoids repeating document or vision computation. It is not a zero-copy
paged-attention scheduler: updated branch caches still allocate memory.

Equal-length buckets avoid introducing padding into sliding-window cache
eviction and rotary positions. Batch size is bounded to limit memory growth.
Static/offloaded caches, full-bidirectional text encoders and non-SDPA attention
are not supported by this path. Context limits apply to document + one branch
+ canvas, with every branch checked before document inference.

Run checkpoint regression tests explicitly with `JIFFY_GPU_TEST=1` and
`python -m unittest discover -s tests -v`. They verify split-token equivalence,
image and 32K execution, batched branch results versus independent prefills,
and alone-versus-group probability stability. Basic fixtures use a 0.03
absolute tolerance. The mixed Score characterization has a separately documented
0.05 bound and does not meet the stricter release-equivalence gate below.
CPU tests verify that cache updates cannot mutate sibling or document caches.

H100 regression timings on 2026-09-21 (single observations, not warmed medians):

| Fixture | Shared document | Separate prefills |
| --- | ---: | ---: |
| 32,042-token text prefix, four equal-length branches | 7.69 s | 29.48 s |
| 32,071-token image/text prefix, two equal-length branches | 8.61 s | 13.13 s |
| Short text, four equal-length branches | 280 ms | 504 ms |
| Small image, two unequal-length branches | 574 ms | 465 ms |

The long fixtures use repeated neutral text, and some branches are duplicates
to exercise batching. These are execution regressions, not natural-document
accuracy benchmarks. Short requests can be slower because splitting the prefix
adds an encoder call. Long documents benefit most from avoiding repeated work.

**Open release issue:** the mixed structured Score fixture shifts its normalized
level probability by 0.0428 versus a monolithic prefill (one underlying binary
match shifts from 0.7423 to 0.6216). Token sequences are identical, and the
isolated Score is identical alone and in the mixed request. Cache inspection
shows exact agreement in the first two layers, tiny differences by layer 5,
and growing differences later, consistent with BF16 numerical sensitivity.
This is not evidence of equal calibration or accuracy. The strict 0.03
probability-equivalence gate is **not passed** for this fixture. Keep the
sequential reference available and run broader accuracy/calibration comparisons
before release; passing characterization tests does not remove this issue.
