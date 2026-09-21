import unittest
import importlib.util

from examples.arcade_games import Survival, Kitchen, Battle, TowerDefense, make_game
from examples.arcade import overlay
from jiffy.protocol import parse_question, json_content


class ArcadeTests(unittest.TestCase):
    def test_contracts_and_render(self):
        for name in ("survival", "kitchen", "battle", "tower"):
            game = make_game(name, 7)
            for question in game.questions.values():
                parse_question(question)
            json_content(game.state(), "state")
            self.assertEqual(game.render().size, (640, 480))
            keys = list(game.questions["action"]["criteria"])
            answer = {"action": {"choice": keys[0], "probabilities": {k: 1/len(keys) for k in keys}}}
            frame = overlay(game.render(), game, answer, 1, 123, 2)
            self.assertEqual(frame.size, (960, 640))

    def test_survival_resources_cannot_be_gathered_twice(self):
        game = Survival(7)
        game.act("gather")
        wood, berries = game.wood, game.inventory_berries
        self.assertGreater(wood, 0)
        game.act("gather")
        self.assertEqual((game.wood, game.inventory_berries), (wood, berries))
        game.wood = 5
        game.act("shelter")
        self.assertTrue(game.success())
        self.assertEqual(game.wood, 0)

    def test_kitchen_complete_recipe(self):
        game = Kitchen(7)
        recipe = ["pantry", "interact", "board", "interact", "stove", "interact", "stove", "stove", "interact", "pass", "interact"]
        for action in recipe * 3:
            game.step({"action": {"choice": action}})
        self.assertEqual(game.score, 3)
        self.assertTrue(game.success())

    def test_battle_energy_and_potions_are_consumed(self):
        game = Battle(7)
        game.act("ice")
        self.assertEqual((game.energy, game.enemy_hp), (1, 106))
        game.act("ice")
        self.assertEqual((game.energy, game.enemy_hp), (1, 106))
        game.act("heal")
        self.assertEqual(game.potions, 1)

    def test_tower_purchases_and_unprotected_loss(self):
        game = TowerDefense(7)
        initial = game.state()
        game.step({"action": {"choice": "wait"}})
        self.assertEqual(initial["enemies"], [])
        snapshot = game.state()
        saved_hp = snapshot["enemies"][0]["hp"]
        game.act("build0")
        self.assertEqual(snapshot["enemies"][0]["hp"], saved_hp)
        self.assertEqual((game.gold, game.towers[0]), (50, 1))
        game.act("build0")
        self.assertEqual(game.gold, 50)
        game = TowerDefense(7)
        for _ in range(80):
            game.step({"action": {"choice": "wait"}})
            if game.done:
                break
        self.assertTrue(game.done)
        self.assertFalse(game.success())

    @unittest.skipUnless(importlib.util.find_spec("Box2D"), "install games extra")
    def test_lander_thruster_signs(self):
        import os
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        angles = {}
        for action in ("left", "right"):
            game = make_game("lander", 20260921)
            try:
                json_content(game.state(), "state")
                game.step({"action": {"choice": action}})
                angles[action] = game.state()["angle"]
            finally:
                game.close()
        self.assertGreater(angles["left"], angles["right"])
