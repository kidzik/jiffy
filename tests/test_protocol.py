import copy
import json
import importlib.util
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import torch
from fastapi.testclient import TestClient

from jiffy.protocol import JevProtocol, json_content, parse_question
from jiffy.server import create_app
from jiffy.diffusion_decisions import compile_contract


class Backend:
    def __init__(self):
        self.calls = []

    def compile(self, questions):
        return questions

    def system_one(self, state, contract, **kwargs):
        self.calls.append((state, contract))
        q = contract["answer"]
        return {"answers": {"answer": q.answer([1 / len(q.options)] * len(q.options))},
                "diagnostics": {"prefix_tokens": 17}}

    def shared_document(self, state, contracts, **kwargs):
        results = [self.system_one(state, c, **kwargs) for c in contracts]
        return {"results": results, "diagnostics": {"input_tokens": 17 * len(results)}}


def request():
    return {"model": "jev-latest", "state": {"message": "Help with billing"}, "questions": {
        "urgent": {"type": "noul", "instructions": {"question": "Urgent?"}},
        "team": {"type": "choice", "instructions": ["Which team?"],
                 "criteria": {"billing": None, "support": {"description": "Technical"}}},
        "severity": {"type": "score", "instructions": "Severity?",
                     "criteria": [{"description": "Low"}, ["High"]]}}}


class ProtocolTests(unittest.TestCase):
    def test_structured_request_and_exact_response(self):
        backend = Backend()
        payload = request()
        result = JevProtocol(backend).evaluate(payload)
        self.assertEqual(set(result), {"model", "answers", "usage"})
        self.assertEqual(result["answers"]["urgent"], {"type": "noul", "noul": .5})
        self.assertEqual(result["usage"], {"input_tokens": 68, "output_tokens": 4})
        self.assertEqual(result["answers"]["severity"]["legend"],
                         {"0": {"description": "Low"}, "1": ["High"]})
        self.assertEqual(result["answers"]["severity"]["score"], .5)
        self.assertEqual(json.loads(backend.calls[0][0]), payload["state"])
        for _, contract in backend.calls:
            self.assertEqual(list(contract), ["answer"])
        levels = [contract["answer"].instructions for _, contract in backend.calls[2:]]
        self.assertNotIn("High", levels[0])
        self.assertNotIn("Low", levels[1])

    def test_renaming_and_adding_questions_preserves_input(self):
        backend = Backend()
        protocol = JevProtocol(backend)
        original = request()
        first = protocol.evaluate(original)["answers"]["urgent"]
        original_call = backend.calls[0]
        alone = copy.deepcopy(original)
        alone["questions"] = {"renamed": alone["questions"]["urgent"]}
        self.assertEqual(protocol.evaluate(alone)["answers"]["renamed"], first)
        self.assertEqual(backend.calls[-1], original_call)

    def test_json_shapes_and_nonfinite_rejection(self):
        for value in ("text", "", {}, [], [{"nested": [True, None, 3]}]):
            self.assertIsInstance(json_content(value, "state"), str)
        for value in (None, 1, False, {1: "x"}, [float("nan")], [float("inf")], [set()]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                json_content(value, "state")

    def test_limits_and_atomic_validation(self):
        backend = Backend()
        payload = request()
        payload["questions"]["team"]["criteria"] = {str(i): None for i in range(255)}
        result = JevProtocol(backend).evaluate(payload)
        self.assertEqual(len(result["answers"]["team"]["probabilities"]), 255)
        backend.calls.clear()
        payload["questions"]["team"]["criteria"]["extra"] = None
        with self.assertRaises(ValueError):
            JevProtocol(backend).evaluate(payload)
        self.assertEqual(backend.calls, [])
        for count in (1, 11):
            with self.assertRaises(ValueError):
                parse_question({"type": "score", "instructions": "?", "criteria": ["a"] * count})
        self.assertEqual(len(parse_question({"type": "score", "instructions": "?", "criteria": ["a"] * 10}).options), 10)

    def test_extended_labels_compile_distinct_tokens(self):
        class Tokenizer:
            pad_token_id = 0

            def get_vocab(self):
                import string
                labels = list(string.ascii_uppercase) + [a + b for a in string.ascii_lowercase for b in string.ascii_lowercase]
                return {label: 1000 + i for i, label in enumerate(labels)}

            def encode(self, text, add_special_tokens=False):
                vocab = self.get_vocab()
                return [vocab[text]] if text in vocab else [ord(c) for c in text]

        question = parse_question({"type": "choice", "instructions": "Pick", "criteria": {str(i): None for i in range(255)}})
        contract = compile_contract(Tokenizer(), {"q": question})
        self.assertEqual(len(set(contract.candidate_ids[0])), 255)


class HttpTests(unittest.TestCase):
    headers = {"Authorization": "Bearer test-only-key"}

    def client(self, backend=None, **kwargs):
        return TestClient(create_app(backend or Backend(), api_key="test-only-key", **kwargs))

    def test_success_and_auth(self):
        with self.client() as client:
            self.assertEqual(client.post("/v1/systemone", json=request()).status_code, 401)
            response = client.post("/v1/systemone", json=request(), headers=self.headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["answers"]["urgent"]["type"], "noul")
            self.assertEqual(client.get("/unknown").status_code, 404)

    @unittest.skipUnless(importlib.util.find_spec("typesafe_sdk"), "install the dev extra for official SDK conformance")
    def test_official_sdk_serialization_and_response_validation(self):
        import httpx2
        from typesafe_sdk import TypeSafeClient, Noul, Choice, Score

        with self.client() as http:
            def dispatch(req):
                response = http.request(req.method, req.url.path, content=req.content, headers=dict(req.headers))
                return httpx2.Response(response.status_code, content=response.content, headers=dict(response.headers))

            with TypeSafeClient(api_key="test-only-key", base_url="http://testserver",
                                transport=httpx2.MockTransport(dispatch)) as client:
                self.assertEqual(client.models.list().models[0].name, "jiffy-diffusiongemma")
                result = client.system_one(state={"message": "Refund please"}, questions={
                    "urgent": Noul(instructions={"question": "Urgent?"}),
                    "team": Choice(instructions=["Which team?"], criteria={"billing": None, "support": ["Bugs"]}),
                    "severity": Score(instructions="Severity?", criteria=[{"description": "Low"}, ["High"]]),
                })
                self.assertEqual(result.nouls["urgent"].noul, .5)
                self.assertEqual(result.scores["severity"].legend[0], {"description": "Low"})
                self.assertEqual(set(result.choices["team"].probabilities), {"billing", "support"})

    def test_validation_and_limits(self):
        with self.client(max_body_bytes=2000) as client:
            for raw in ('{', '{"model":"jev-latest","model":"jev-latest"}', '{}', 'x' * 2001):
                response = client.post("/v1/systemone", content=raw, headers=self.headers)
                self.assertEqual(response.status_code, 422)
        with self.client(requests_per_minute=1) as client:
            self.assertEqual(client.post("/v1/systemone", json=request(), headers=self.headers).status_code, 200)
            self.assertEqual(client.post("/v1/systemone", json=request(), headers=self.headers).status_code, 429)

    def test_busy_and_recovery(self):
        entered, release = threading.Event(), threading.Event()

        class SlowBackend(Backend):
            def system_one(self, *args, **kwargs):
                entered.set()
                if not release.wait(10):
                    raise RuntimeError("test timed out")
                return super().system_one(*args, **kwargs)

        with self.client(SlowBackend()) as client, ThreadPoolExecutor(1) as pool:
            future = pool.submit(client.post, "/v1/systemone", json=request(), headers=self.headers)
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(client.post("/v1/systemone", json=request(), headers=self.headers).status_code, 529)
            finally:
                release.set()
            self.assertEqual(future.result(timeout=10).status_code, 200)
            self.assertEqual(client.post("/v1/systemone", json=request(), headers=self.headers).status_code, 200)

    def test_oom_is_overload_not_internal_error(self):
        class BrokenBackend(Backend):
            def system_one(self, *args, **kwargs):
                raise torch.cuda.OutOfMemoryError("private details")

        with self.client(BrokenBackend()) as client:
            response = client.post("/v1/systemone", json=request(), headers=self.headers)
            self.assertEqual(response.status_code, 529)
            self.assertNotIn("private", response.text)


if __name__ == "__main__":
    unittest.main()
