# Jiffy

**Preset typed questions. Changing documents and images. All answer distributions together.**

Jiffy uses frozen DiffusionGemma to answer `noul` (yes/no), `choice` (named
alternatives), and `score` (ordered rubric) questions. By default, one encoder
prefill is followed by one parallel decoder pass for every answer position.
No task-specific training or free-form text generation is required.

**Experimental.** Probabilities are uncalibrated conditional scores, not proven
joint marginals. Jiffy is an independent Jev-like alternative, not affiliated
with TypeSafe AI or a drop-in implementation of its API. Broad accuracy and
calibration parity have not been established. Code licensing remains pending;
the upstream model has its own license. This repository is not a public service.

## Install

Validated inference hardware: Linux, Python 3.10, NVIDIA H100 80GB, BF16.

```bash
git clone https://github.com/kidzik/jiffy.git
cd jiffy
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
jiffy --help
```

All model-runtime dependencies are pinned. Install a matching CUDA PyTorch wheel
for your host if needed. Initial inference downloads Google's checkpoint and
processor at revision `f7f5b7f5fa82ffc52addd066915886d497f5517b`.
Model weights, datasets and API credentials are not included.

## Quickstart

```bash
jiffy examples/diffusion-decisions.json --output /tmp/jiffy-result.json
```

The example covers all three question types. Use a new output path for each run.
`python -m jiffy` and `jiffy-decide` expose the same CLI.

For repeated documents, retain the model and compile the questions once:

```python
from jiffy import DiffusionDecisions, Question, image_data_url

model = DiffusionDecisions.from_backbone(image_tokens=560)
contract = model.compile({
    "paid": Question.noul("Has this invoice already been paid?"),
    "currency": Question.choice("Which currency is explicitly stated?", {
        "usd": "US dollars (USD)", "eur": "Euros (EUR)", "unknown": "Not stated",
    }),
})
result = model.system_one({
    "text": "Review this invoice.",
    "images": [image_data_url("invoice.png")],
}, contract)
print(result["answers"])
```

For text-only inputs, pass a string as the state. Images must be embedded PNG,
JPEG or WebP data URLs; `image_data_url` explicitly reads a client-side file.
Questions still enter the encoder on every request: compilation reuses the
prompt/scaffold, not offline neural question representations.

## Contract and Limits

- Every answer includes `type` and the complete `probabilities` distribution.
- Noul returns yes probability in `noul`; Choice returns the selected key in `choice`.
- Score returns the expected zero-based rubric index in `score`, plus the rubric legend.
- Confidence is one minus normalized entropy, not calibrated correctness probability.
- 1-16 questions; 2-26 choices; Score has 2-10 levels.
- At most four images, 16 million pixels each, with encoded-input limits enforced.
- 32,768 total positions including the 256-token answer canvas; no silent truncation.
- One decoder step is the default. More steps are experimental, not a quality guarantee.
- Missing evidence needs an explicit Unknown option or a separate evidence question.
- Single-request local SDK/CLI only; concurrent serving, hosted authentication, CPU
  inference, video and arbitrary object/array states are not supported promises.

## Measured Evidence

On a single H100, the small image pilot measured about **251 ms** median for
short requests and **6.64 seconds** near 32K. Public-image accuracy was **19/22
decisions on 11 images**. A separate structured-state Doom test achieved 15 kills
over three seeds, zero exits, and 174 ms median on 96 saved states. Doom was not
screen-based. These are feasibility results, not broad generalization claims.
See [evaluation notes](docs/evaluation.md) and [release readiness](docs/release.md).

## Development

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m build
```

Tests use local fixtures and do not download model weights. The CPU CI template
in `ci/github-actions.yml` is not yet activated; it checks contracts and packaging.
A target-GPU smoke is needed to validate actual inference.
The package is standalone: no dependency on another Jiffy repository, Qwen,
training code, Doom, or benchmark artifacts.
