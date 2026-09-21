# Release Readiness

Standalone experimental SDK/CLI. Nothing here is a hosted service or a promise
of Jev accuracy/calibration parity. The repository is private pending release.

Extraction verification: all 14 local tests passed and the standalone wheel
built successfully. Installed into a temporary directory outside either checkout,
the wheel answered the portable three-type request correctly on H100 in one
decoder step. This reused installed pinned dependencies and cached upstream
weights, not source from another package. Fresh-machine installation remains
unverified. The current GitHub credential cannot create Actions workflows;
`ci/github-actions.yml` is the ready-to-enable template. Move it to
`.github/workflows/ci.yml` with a workflow-authorized token before relying on CI.

- [x] DiffusionGemma-only dependencies, source, examples and tests.
- [x] Pinned checkpoint and processor revision.
- [x] Full candidate distributions and one-step default.
- [ ] Owner-approved code license and upstream notice review.
- [ ] Activate the supplied GitHub Actions template with a workflow-authorized token.
- [ ] Fresh-machine installation and target-GPU image/32K release smoke.
- [ ] Broader held-out natural-document/image evaluation.
- [ ] Missing/conflicting evidence, unreadable images and prompt-injection evaluation.
- [ ] Maximum question/option count and repeated mixed-length memory testing.
- [ ] Calibration and abstention evaluation before calibrated-probability claims.
- [ ] Publication approval and clean release tag.

The backend uses private Transformers cache/decoder interfaces. Runtime changes
require numerical regression testing. GPU access should be serialized by callers;
the backend is not a concurrent request scheduler.

Upstream model: [Google DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it).
Weights are downloaded separately. Historical research and unrelated model
implementations were intentionally not migrated into this repository.
