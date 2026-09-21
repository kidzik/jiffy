import unittest
from collections import deque
from types import SimpleNamespace

from examples.doom import QUESTIONS, VARIABLES, command, observe, positive
from jiffy.protocol import parse_question


class DoomDemoTests(unittest.TestCase):
    def test_contract(self):
        for question in QUESTIONS.values():
            parse_question(question)

    def test_buttons(self):
        answers = {"movement": {"choice": "forward"}, "turn": {"choice": "right"},
                   "fire": {"noul": .8}, "use": {"noul": .1}}
        self.assertEqual(command(answers), [True, False, False, False, False, True, True, False])
        answers["fire"]["noul"] = float("nan")
        with self.assertRaises(ValueError):
            command(answers)

    def test_invalid_movement(self):
        with self.assertRaises(ValueError):
            command({"movement": {"choice": "teleport"}, "turn": {"choice": "none"}})

    def test_positive(self):
        self.assertEqual(positive("8"), 8)
        with self.assertRaises(ValueError):
            positive("abc")

    def test_blocked_and_stalled_observation(self):
        import numpy as np
        frame = SimpleNamespace(labels=[], depth_buffer=np.ones((240, 320)))
        game = SimpleNamespace(get_state=lambda: frame, get_game_variable=lambda key: 0.)
        vzd = SimpleNamespace(GameVariable=SimpleNamespace(**{key: key for key in VARIABLES}))
        history = deque([{"x": 0, "y": 0}] * 3)
        result = observe(game, vzd, history)
        self.assertEqual(result["navigation"], {"blocked_ahead": True, "stalled": True})
        frame.depth_buffer[:] = 100
        result = observe(game, vzd, deque())
        self.assertEqual(result["navigation"], {"blocked_ahead": False, "stalled": False})
