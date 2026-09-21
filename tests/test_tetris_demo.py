import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("tetris"), "install tetris extra")
class TetrisTests(unittest.TestCase):
    def test_preview_isolation_and_piece_sequence(self):
        from examples.tetris_match import new_game, placements, request
        from jiffy.protocol import json_content, parse_question
        a, b = new_game(123), new_game(123)
        before = a.board.tobytes()
        queue = list(a.queue)
        choices = placements(a)
        self.assertGreater(len(choices), 1)
        self.assertEqual(a.board.tobytes(), before)
        self.assertEqual(list(a.queue), queue)
        for c in choices.values():
            self.assertEqual(c["game"].piece.type, queue[0])
            self.assertIsNot(c["game"].board, a.board)
            self.assertIs(c["game"].gravity.game, c["game"])
            self.assertIs(c["game"].rs.board, c["game"].board)
        selected = next(iter(choices.values()))["game"]
        selected.board[-1, 0] = 7
        self.assertEqual(a.board.tobytes(), before)
        payload = request(a, choices)
        json_content(payload["state"], "state")
        parse_question(payload["questions"]["placement"])
        self.assertEqual(a.piece.type, b.piece.type)

    def test_engine_line_clear(self):
        from examples.tetris_match import new_game, placements
        import tetris
        game = new_game(1)
        game.piece = game.rs.spawn(tetris.PieceType.I)
        for column in range(10):
            game.board[-1, column] = 0 if 3 <= column < 7 else 1
        choices = placements(game)
        self.assertTrue(any(c["facts"]["lines_cleared"] == 1 for c in choices.values()))

    def test_holes_and_height(self):
        import numpy as np
        from examples.tetris_match import features
        state = features(np.array([[0, 0], [1, 0], [0, 1]]))
        self.assertEqual(state["holes"], 1)
        self.assertEqual(state["column_heights"], [2, 1])
