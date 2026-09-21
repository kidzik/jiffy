# Release Status

Version 0.1.0 is an **experimental prerelease**, not a calibrated or production
equivalent of TypeSafe Jev. Code is Apache-2.0; model weights are downloaded
separately under Google's own terms.

## Verified

- Standalone DiffusionGemma package, CLI, and authenticated local HTTP adapter.
- Pinned model revision and runtime dependencies.
- CPU contract, validation, cache-isolation, and HTTP tests.
- Official TypeSafe Python SDK interoperability tests.
- H100 checks covering images, 32K inputs, and 255-option inference.
- Public JevBench evaluation: 195/231 correct, all responses valid.
- Fresh standalone dependency installation; `pip check` passes.

The compatibility report documents the workloads and numerical checks.
Characterization tests do not imply model equivalence: a mixed Score fixture
shifted by 0.0428 versus the sequential reference, above the stricter 0.03
equivalence threshold. This is an acknowledged prerelease limitation.

## Before a Production Release

- Broader held-out document/image accuracy and calibration evaluation.
- Resolve or explicitly accept shared-prefix numerical drift against those data.
- Prompt-injection, missing evidence, unreadable image, and abstention evaluation.
- Maximum-size and repeated mixed-length memory/load testing.
- Deployment-specific TLS, access control, observability, and request scheduling.
- Validation beyond H100; quantized deployment is not supported.

The backend uses private pinned Transformers interfaces. Runtime upgrades need
numerical regression testing. Local SDK callers must serialize GPU access; the
HTTP server enforces one active GPU request and rejects overlap.

Optional GPU tests require the checkpoint and a suitable CUDA environment.
CPU CI does not download weights or establish GPU accuracy or performance.
The workflow remains an inactive template in `ci/github-actions.yml` until a
credential with workflow-write permission activates it.
