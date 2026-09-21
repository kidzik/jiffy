import unittest

import torch

from jiffy.diffusion_decisions import DiffusionDecisions, SlotConstraints, SlotStopping, answer_scores, compile_contract
from jiffy.schema import Question


class Tokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class DiffusionDecisionTests(unittest.TestCase):
    def test_typed_contract_has_distinct_slots(self):
        questions = {"urgent": Question.noul("Urgent?"), "team": Question.choice("Team?", {"x": "Billing", "y": "Support"}),
                     "severity": Question.score("Severity?", ["Low", "Medium", "High"])}
        contract = compile_contract(Tokenizer(), questions)
        self.assertEqual(contract.names, tuple(questions))
        self.assertEqual(len(set(contract.positions)), 3)
        self.assertEqual(len(contract.scaffold), 256)
        self.assertIn("A: The answer is no.", contract.prompt)
        self.assertIn("C: High", contract.prompt)

    def test_bad_contracts_are_rejected(self):
        q = Question.noul("Test?")
        for questions in ({}, {"": q}, {"q": "not typed"}, {str(i): q for i in range(17)}):
            with self.assertRaises(ValueError):
                compile_contract(Tokenizer(), questions)
        with self.assertRaises(ValueError):
            compile_contract(Tokenizer(), {"q": q}, orders={"q": [0, 0]})
        with self.assertRaises(ValueError):
            compile_contract(Tokenizer(), {"q": q}, canvas_length=2)

    def test_constraints_keep_all_candidate_probabilities(self):
        contract = compile_contract(Tokenizer(), {"q": Question.noul("Test?")}, canvas_length=64)
        scores = torch.zeros(1, 64, 128)
        position = contract.positions[0]
        scores[0, position, ord("B")] = torch.log(torch.tensor(3.))
        constrained = SlotConstraints(contract)
        output = constrained(torch.zeros(1, 1), scores, 4)
        self.assertEqual(constrained.trace[0]["step"], 1)
        torch.testing.assert_close(torch.tensor(constrained.trace[0]["answers"][0]["probabilities"]), torch.tensor([.25, .75]))
        self.assertLess(constrained.trace[0]["answers"][0]["allowed_mass"], .04)
        self.assertEqual(torch.isfinite(output[0, position]).sum().item(), 2)
        self.assertEqual(torch.isfinite(output[0, 0]).sum().item(), 1)
        self.assertTrue(torch.isfinite(scores).all())

    def test_option_permutation_restores_original_semantics(self):
        contract = compile_contract(Tokenizer(), {"q": Question.noul("Test?")}, canvas_length=64, orders={"q": [1, 0]})
        scores = torch.zeros(1, 64, 128)
        scores[0, contract.positions[0], ord("B")] = torch.log(torch.tensor(3.))
        constraint = SlotConstraints(contract)
        constraint(torch.zeros(1, 1), scores, 1)
        torch.testing.assert_close(torch.tensor(constraint.trace[-1]["answers"][0]["probabilities"]), torch.tensor([.75, .25]))

    def test_selected_readout_matches_full_canvas_extraction(self):
        contract = compile_contract(Tokenizer(), {"q": Question.noul("Test?")}, canvas_length=64)
        scores = torch.randn(1, 64, 128)
        constraint = SlotConstraints(contract)
        constraint(torch.zeros(1, 1), scores, 1)
        self.assertEqual(answer_scores(scores[:, list(contract.positions)], contract), constraint.trace[-1]["answers"])

    def test_prefix_is_bound_to_backend_and_contract(self):
        backend = object.__new__(DiffusionDecisions)
        contract = compile_contract(Tokenizer(), {"q": Question.noul("Test?")})
        with self.assertRaisesRegex(ValueError, "foreign"):
            backend.score({"owner": object(), "contract": contract}, contract)
        with self.assertRaisesRegex(ValueError, "mismatched"):
            backend.score({"owner": backend, "contract": None}, contract)
        with self.assertRaisesRegex(ValueError, "positive"):
            backend.score({}, contract, steps=0)

    def test_padding_cannot_fake_early_confidence(self):
        stopping = SlotStopping([7])
        canvas = torch.zeros(1, 256, dtype=torch.long)
        scores = torch.full((1, 256, 2), -torch.inf)
        scores[:, :, 0] = 0
        scores[:, 7, 1] = 0
        self.assertFalse(stopping(canvas, scores).item())
        self.assertFalse(stopping(canvas, scores).item())
        scores[:, 7, 1] = -100
        self.assertTrue(stopping(canvas, scores).item())


if __name__ == "__main__":
    unittest.main()
