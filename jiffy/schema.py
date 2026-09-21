"""Typed questions, evidence inputs, and probability validation."""
from dataclasses import dataclass
import math


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def validate_state(value):
    """Text or embedded images; never interpret server input as paths or URLs."""
    if isinstance(value, str):
        return text(value, "state")
    if not isinstance(value, dict) or set(value) - {"text", "images"}:
        raise ValueError("state must be text or an object with text and images")
    content, images = value.get("text", ""), value.get("images", [])
    if not isinstance(content, str) or not isinstance(images, list) or len(images) > 4:
        raise ValueError("state needs string text and at most four images")
    if not content.strip() and not images:
        raise ValueError("state must contain text or images")
    for image in images:
        if (not isinstance(image, str) or len(image) > 24_000_000 or not image.startswith(
                ("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,"))):
            raise ValueError("images must be PNG/JPEG/WebP base64 data URLs under 24 MB")
        import base64
        try:
            if not base64.b64decode(image.split(",", 1)[1], validate=True):
                raise ValueError("empty image")
        except ValueError as error:
            raise ValueError("invalid base64 image") from error
    return value


@dataclass(frozen=True)
class Option:
    key: str
    description: str


@dataclass(frozen=True)
class Question:
    kind: str
    instructions: str
    options: tuple[Option, ...]

    def __post_init__(self):
        text(self.instructions, "instructions")
        limits = {"choice": (2, 255), "score": (2, 10), "noul": (2, 2)}
        if self.kind not in limits:
            raise ValueError("type must be choice, score, or noul")
        object.__setattr__(self, "options", tuple(self.options))
        lo, hi = limits[self.kind]
        if not lo <= len(self.options) <= hi:
            raise ValueError(f"{self.kind} needs {lo}–{hi} options")
        for option in self.options:
            text(option.key, "option key")
            text(option.description, "option description")
        if len({o.key for o in self.options}) != len(self.options):
            raise ValueError("option keys must be unique")
        if self.kind == "noul" and tuple(o.key for o in self.options) != ("false", "true"):
            raise ValueError("Noul requires false, true in that order")

    @classmethod
    def choice(cls, instructions, criteria):
        return cls("choice", instructions, tuple(Option(k, v) for k, v in criteria.items()))

    @classmethod
    def score(cls, instructions, criteria):
        return cls("score", instructions, tuple(Option(str(i), v) for i, v in enumerate(criteria)))

    @classmethod
    def noul(cls, instructions, false="The answer is no.", true="The answer is yes."):
        return cls("noul", instructions, (Option("false", false), Option("true", true)))

    @classmethod
    def from_dict(cls, data):
        kind, instructions = data["type"], data["instructions"]
        if kind == "choice":
            return cls.choice(instructions, data["criteria"])
        if kind == "score":
            if not isinstance(data["criteria"], list):
                raise ValueError("Score criteria must be an ordered list")
            return cls.score(instructions, data["criteria"])
        if kind == "noul":
            criteria = data.get("criteria", {"false": "The answer is no.", "true": "The answer is yes."})
            if set(criteria) != {"false", "true"}:
                raise ValueError("Noul criteria must contain false and true")
            return cls.noul(instructions, criteria["false"], criteria["true"])
        raise ValueError("unsupported question type")

    def to_dict(self):
        criteria = ([o.description for o in self.options] if self.kind == "score"
                    else {o.key: o.description for o in self.options})
        return {"type": self.kind, "instructions": self.instructions, "criteria": criteria}

    def answer(self, probabilities):
        p = validate_probabilities(probabilities, len(self.options))
        if self.kind == "noul":
            return {"type": "noul", "noul": p[1], "probabilities": {"false": p[0], "true": p[1]}}
        # Educational convention; TypeSafe does not document this exact formula.
        entropy = -sum(x * math.log(x) for x in p if x > 0)
        answer = {"type": self.kind,
                  "probabilities": {o.key: x for o, x in zip(self.options, p)},
                  "confidence": max(0.0, min(1.0, 1 - entropy / math.log(len(p))))}
        if self.kind == "choice":
            answer["choice"] = self.options[max(range(len(p)), key=p.__getitem__)].key
        else:
            answer["score"] = sum(i * x for i, x in enumerate(p))
            answer["legend"] = {o.key: o.description for o in self.options}
        return answer


def validate_probabilities(values, n):
    if (not isinstance(values, (list, tuple)) or len(values) != n
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 or v > 1 for v in values)
            or not math.isclose(sum(values), 1.0, abs_tol=1e-5)):
        raise ValueError(f"expected {n} finite probabilities summing to one")
    return list(map(float, values))
