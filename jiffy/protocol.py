"""TypeSafe-shaped requests with isolated questions and honest model identity."""
import copy
import json

from .schema import Question, text
from .diffusion_decisions import MODEL

MODEL_ALIAS = "jiffy-diffusiongemma"


def json_content(value, name):
    if not isinstance(value, (str, dict, list)):
        raise ValueError(f"{name} must be a string, object, or array")
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(k, str) for k in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValueError("invalid JSON value")
    check(value)
    rendered = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return value if isinstance(value, str) and value.strip() else rendered


def parse_question(value):
    if not isinstance(value, dict) or set(value) - {"type", "instructions", "criteria"}:
        raise ValueError("invalid question fields")
    kind = value["type"]
    instructions = json_content(value["instructions"], "instructions")
    criteria = value.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict):
            raise ValueError("choice criteria must be an object")
        descriptions = {}
        for key, description in criteria.items():
            text(key, "choice key")
            descriptions[key] = key if description is None else key + ": " + json_content(description, "criterion")
        return Question.choice(instructions, descriptions)
    if kind == "score":
        if not isinstance(criteria, list):
            raise ValueError("score criteria must be an array")
        return Question.score(instructions, [json_content(c, "level") for c in criteria])
    if kind == "noul":
        if "criteria" not in value:
            return Question.noul(instructions)
        if not isinstance(criteria, dict) or set(criteria) - {"false", "true"}:
            raise ValueError("noul criteria must contain only false/true")
        return Question.noul(instructions,
            json_content(criteria.get("false", "The answer is no."), "false criterion"),
            json_content(criteria.get("true", "The answer is yes."), "true criterion"))
    raise ValueError("unsupported question type")


class JevProtocol:
    """Question isolation with a shared document cache, or a sequential reference.

    API aliases never change the returned identity to impersonate Jev.
    The private Jev confidence formula and calibration are not reproduced.
    """
    def __init__(self, backend, max_questions=256, *, execution="shared_document"):
        if type(max_questions) is not int or max_questions < 1:
            raise ValueError("max_questions must be positive")
        self.backend, self.max_questions = backend, max_questions
        if execution not in ("shared_document", "sequential"):
            raise ValueError("execution must be shared_document or sequential")
        self.execution = execution

    def evaluate(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"model", "state", "questions"}:
            raise ValueError("provide exactly model, state, and questions")
        if payload["model"] not in (MODEL_ALIAS, MODEL, "jev-latest"):
            raise ValueError("unsupported model; use jiffy-diffusiongemma")
        state = json_content(payload["state"], "state")
        definitions = payload["questions"]
        if not isinstance(definitions, dict) or not 1 <= len(definitions) <= self.max_questions:
            raise ValueError(f"provide 1-{self.max_questions} questions")
        parsed = {}
        for name, definition in definitions.items():
            text(name, "question id")
            parsed[name] = parse_question(definition)
        # Validate all contracts before any inference. No partial responses.
        contracts = {}
        for name, q in parsed.items():
            evaluations = [q]
            if q.kind == "score":
                evaluations = [Question.noul(
                    "Evaluate whether the supplied state matches the following rubric description "
                    "for this question.\nQuestion: " + q.instructions + "\nDescription: " + option.description)
                    for option in q.options]
            contracts[name] = [self.backend.compile({"answer": item}) for item in evaluations]
        flat = [contract for evaluations in contracts.values() for contract in evaluations]
        if self.execution == "shared_document":
            batch = self.backend.shared_document(state, flat, seed=20260921)
            flat_results = batch["results"]
            input_tokens = batch["diagnostics"]["input_tokens"]
        else:
            flat_results = [self.backend.system_one(state, c, steps=1, seed=20260921) for c in flat]
            input_tokens = sum(r["diagnostics"]["prefix_tokens"] for r in flat_results)
        if len(flat_results) != len(flat):
            raise RuntimeError("incomplete branch results")
        answers, output_tokens, cursor = {}, 0, 0
        for name, evaluations in contracts.items():
            results = flat_results[cursor:cursor + len(evaluations)]
            cursor += len(evaluations)
            output_tokens += len(results)
            question = parsed[name]
            if question.kind == "score":
                weights = [result["answers"]["answer"]["noul"] for result in results]
                total = sum(weights)
                probabilities = [w / total for w in weights] if total else [1 / len(weights)] * len(weights)
                answer = question.answer(probabilities)
                answer["legend"] = {str(i): copy.deepcopy(value)
                                    for i, value in enumerate(definitions[name]["criteria"])}
            else:
                answer = copy.deepcopy(results[0]["answers"]["answer"])
            if answer["type"] == "noul":
                answer = {"type": "noul", "noul": answer["noul"]}
            answers[name] = answer
        # There is no generated text: output_tokens counts scored answer slots.
        return {"model": MODEL, "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}
