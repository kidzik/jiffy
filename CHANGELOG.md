# Changelog

## 0.1.0 - 2026-09-21

Experimental public release.

- Frozen DiffusionGemma typed decisions: Noul, Choice, and Score.
- Local SDK and CLI for text and embedded images, with a 32K context limit.
- Authenticated TypeSafe-shaped HTTP API and model discovery.
- Shared document encoding with isolated question/Score-level cache branches.
- Complete candidate distributions and support for 255-choice contracts.
- Official TypeSafe SDK interoperability checks and opt-in H100 regressions.
- Reproducible evaluation on the 231 public JevBench tasks: 195 correct.

Known limitations: probabilities are uncalibrated; splitting prefills can
change scores; short workloads can be slower with cache branching. GPU
validation is limited to H100. No hosted service, quantized runtime, CPU/Apple
GPU inference, or Jev accuracy/calibration parity is promised.
