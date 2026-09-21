# Jiffy

Typed decisions over documents and images with frozen DiffusionGemma.
Preset your questions, change the evidence, and receive bounded answers with
complete probability distributions. No task-specific training is required.

**Experimental v0.1.0.** Jiffy is an independent Jev-style implementation, not
affiliated with TypeSafe AI. API interoperability is tested; equal accuracy,
calibration, and latency are not promised. Probabilities are uncalibrated.

## Install

Validated runtime: Linux, Python 3.10, CUDA, NVIDIA H100 80GB, BF16.

```bash
git clone https://github.com/kidzik/jiffy.git
cd jiffy
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[serve]'
```

The model and processor are downloaded at pinned revision
`f7f5b7f5fa82ffc52addd066915886d497f5517b` of
[Google DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it).
Weights and credentials are not included. CPU, Apple GPU, quantized inference,
and multi-GPU sharding are not supported by this implementation.

## Jev-Style API

```bash
export JIFFY_API_KEY='<choose-a-local-secret>'
jiffy-serve --host 127.0.0.1 --port 8000
```

Call authenticated `POST /v1/systemone` with `model`, `state`, and `questions`.
Model discovery is available at `GET /v1/models`. Use
`model: "jiffy-diffusiongemma"`; `jev-latest` is a migration alias only, and
responses identify the actual Google model.

```python
from jiffy import DiffusionDecisions, JevProtocol

model = DiffusionDecisions.from_backbone()
api = JevProtocol(model)
result = api.evaluate({
    "model": "jiffy-diffusiongemma",
    "state": {"message": "Please refund the duplicate charge."},
    "questions": {
        "refund": {"type": "noul", "instructions": "Is a refund requested?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Payments and refunds", "support": "Software bugs"},
        },
    },
})
print(result["answers"])
```

The adapter supports structured JSON states, instructions, and criteria; Noul,
Choice (2-255 options), and Score (2-10 levels); and complete distributions.
It encodes the document once and forks isolated question caches. Equal-length
branches run in batches of up to four; other branches run separately without
re-encoding the document. Score levels are evaluated independently.

`JevProtocol(model, execution="sequential")` retains the separate-prefill
reference. Read the [compatibility contract](docs/compatibility.md) for response
shapes, errors, usage accounting, limits, and known differences.

## Images and Shared-Pass SDK

HTTP states follow the TypeSafe JSON contract. Image attachments are a separate
local SDK extension:

```python
from jiffy import DiffusionDecisions, Question, image_data_url

model = DiffusionDecisions.from_backbone(image_tokens=560)
contract = model.compile({
    "paid": Question.noul("Has this invoice already been paid?"),
    "currency": Question.choice("Which currency is explicitly stated?", {
        "usd": "US dollars", "eur": "Euros", "unknown": "Not stated",
    }),
})
result = model.system_one({
    "text": "Review this invoice.",
    "images": [image_data_url("invoice.png")],
}, contract)
print(result["answers"])
```

This fast SDK mode uses one shared encoder prefill and one decoder pass. Its
questions share attention and are **not isolated**. For isolated image decisions,
pass single-question compiled contracts to
`model.shared_document(state, contracts)` instead.

The SDK supports 1-16 questions per shared canvas, up to four embedded PNG/JPEG/
WebP images, and 32,768 total positions including the 256-token canvas. It
rejects oversized inputs rather than truncating. Compilation reuses the prompt
and scaffold, not offline neural representations of the questions.

CLI example (use a new output path):

```bash
jiffy examples/diffusion-decisions.json --output result.json
```

`python -m jiffy` and `jiffy-decide` expose the same CLI.

## Evidence and Hardware

On the unchanged public JevBench subset, Jiffy scored **195/231 (84.4%)**.
Published Jev scored 200/231 on the same IDs. Local median/p95 latency was
259/588 ms. Missing private/imported tasks and an unknown hosting tariff mean
this is not a full-suite score or official ranking.
[Evaluation details and reproduction](docs/jevbench-public.md).

Observed BF16 GPU footprint on H100 was about **50 GiB** for a small image with
four branches and **69 GiB** for 32K plus an image with four branches. Host RAM
peaked near 49 GiB during loading. Consumer-GPU quantization is future work.

Known limitations include hard-tier calibration error (ECE 0.204), numerical
drift when splitting prefills, and slower short requests when sharing cannot
amortize its overhead. See [release status](docs/release.md).

## Development

```bash
python -m pip install -e '.[dev,serve]'
python -m unittest discover -s tests -v
python -m build
```

Default tests use CPU fixtures and do not download weights. Opt into checkpoint
regressions with `JIFFY_GPU_TEST=1`. Run benchmark commands from the repository
root; the harness is developer tooling, not part of the runtime wheel.
The CPU CI template at `ci/github-actions.yml` is not active yet. Move it to
`.github/workflows/ci.yml` using a credential with workflow-write permission.

## License

Code is [Apache-2.0](LICENSE). Upstream weights, dependencies, and external
benchmark assets retain their respective licenses; see [NOTICE](NOTICE).
