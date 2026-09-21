import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jiffy import Question, image_data_url
from jiffy.io import atomic_json
from jiffy.media import decode_image
from jiffy.schema import validate_state, validate_probabilities


class ApiTests(unittest.TestCase):
    def test_lightweight_import(self):
        subprocess.run([sys.executable, "-c", "import sys; from jiffy import Question; assert 'torch' not in sys.modules"], check=True)

    def test_public_backend(self):
        from jiffy import DiffusionDecisions
        self.assertTrue(callable(DiffusionDecisions.from_backbone))

    def test_example_contract(self):
        request = json.loads((Path(__file__).resolve().parents[1] / "examples/diffusion-decisions.json").read_text())
        validate_state(request["state"])
        self.assertEqual({Question.from_dict(q).kind for q in request["questions"].values()}, {"choice", "noul", "score"})

    def test_roundtrip_and_answers(self):
        for q in [Question.noul("Paid?"), Question.choice("Currency?", {"usd": "USD", "eur": "EUR"}),
                  Question.score("Size?", ["Small", "Large"])]:
            self.assertEqual(Question.from_dict(q.to_dict()), q)
            self.assertAlmostEqual(sum(q.answer([.25, .75])["probabilities"].values()), 1.)
        self.assertEqual(Question.score("Size?", ["Small", "Large"]).answer([.25, .75])["score"], .75)

    def test_invalid_inputs(self):
        for state in ["", {}, {"images": ["https://example.com/x.png"]}, {"path": "/tmp/x.png"}]:
            with self.assertRaises(ValueError):
                validate_state(state)
        for probabilities in [[1., 1.], [float("nan"), 0.], [-1., 2.]]:
            with self.assertRaises(ValueError):
                validate_probabilities(probabilities, 2)

    def test_image_roundtrip(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.png"
            Image.new("RGB", (12, 8), "red").save(path)
            url = image_data_url(path)
            validate_state({"images": [url]})
            image = decode_image(url)
            self.assertEqual(image.size, (12, 8))
            self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))

    def test_atomic_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            atomic_json(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})
            with self.assertRaises(ValueError):
                atomic_json(path, {"bad": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
