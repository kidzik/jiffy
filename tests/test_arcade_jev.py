import unittest
import httpx

from examples.compare_arcade import Jev


class JevArcadeTests(unittest.TestCase):
    request = {"model": "jiffy-diffusiongemma", "state": {"health": 10},
               "questions": {"action": {"type": "choice", "instructions": "Choose", "criteria": {"a": "A", "b": "B"}}}}

    def client(self, result=None, status=200, max_calls=2):
        result = result or {"model": "jev-1.13.0", "answers": {"action": {"type": "choice", "choice": "a", "probabilities": {"a": .8, "b": .2}}}}
        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status, json=result)))
        return Jev("test-not-a-real-key", max_calls=max_calls, client=client)

    def test_validation_and_call_cap(self):
        api = self.client(max_calls=1)
        self.assertEqual(api.evaluate(self.request)["answers"]["action"]["choice"], "a")
        self.assertEqual(api.resolved, "jev-1.13.0")
        with self.assertRaisesRegex(RuntimeError, "cap"):
            api.evaluate(self.request)
        api.close()

    def test_auth_failure_has_no_fallback_or_secret(self):
        api = self.client(status=401)
        with self.assertRaisesRegex(RuntimeError, "TypeSafe HTTP 401; no fallback"):
            api.evaluate(self.request)
        self.assertEqual(api.calls, 1)
        api.close()

    def test_invalid_distribution_rejected(self):
        api = self.client({"model": "jev-1.13.0", "answers": {"action": {"type": "choice", "choice": "a", "probabilities": {"a": .8, "b": .8}}}})
        with self.assertRaisesRegex(ValueError, "distribution"):
            api.evaluate(self.request)
        api.close()

    def test_rounded_distribution_preserved(self):
        api = self.client({"model": "jev-1.13.0", "answers": {"action": {"type": "choice", "choice": "a", "probabilities": {"a": .8, "b": .21}}}})
        self.assertEqual(api.evaluate(self.request)["answers"]["action"]["probabilities"]["b"], .21)
        api.close()
