"""Two Gymnasium environments and four original, small demonstration games.

Only environment dynamics live here. Model decisions are made by arcade.py.
"""
import math
import random

from PIL import Image, ImageDraw, ImageFont


def font(size=18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def label(draw, xy, value, size=18, color="white"):
    draw.text(xy, str(value), font=font(size), fill=color)


def bar(draw, x, y, w, value, color):
    draw.rounded_rectangle((x, y, x+w, y+10), radius=3, fill="#28333c")
    if value > 0:
        draw.rounded_rectangle((x, y, x+max(3, w*min(1, value)), y+10), radius=3, fill=color)


def choice(instructions, options):
    return {"type": "choice", "instructions": instructions, "criteria": options}


class Physics:
    def __init__(self, name, seed):
        import gymnasium as gym
        self.name = name
        self.title = "APEX / Racing" if name == "racing" else "DESCENT / Lunar Lander"
        self.env = gym.make("CarRacing-v3" if name == "racing" else "LunarLander-v3",
                            render_mode="rgb_array", continuous=name == "racing")
        self.obs, _ = self.env.reset(seed=seed)
        self.done, self.reward, self.tick = False, 0., 0
        self.last_reward = 0.
        self.repeat = 8 if name == "racing" else 4
        self.fps = 50
        self.limit = 180 if name == "racing" else 140
        if name == "racing":
            self.questions = {
                "steer": choice("Drive forward along the road. Choose steering from the upcoming road bearing in car-local coordinates: negative bearing means left; positive means right. Keep centered. Do not continuously turn when the bearing is near zero.",
                                {"left": "Steer left", "straight": "No steering", "right": "Steer right"}),
                "pedal": choice("Make progress along the racetrack without leaving the road. Accelerate from rest; coast or brake for sharp bends at high speed.",
                                {"gas": "Accelerate", "coast": "Coast", "brake": "Brake"}),
            }
        else:
            self.questions = {"action": choice(
                "Land upright and gently on the pad at x=0, y=0. Observation is normalized: x positive is right of pad; y is altitude; vy negative is falling; angle positive is counterclockwise. Main engine slows descent. Action left increases angle and pushes left; action right decreases angle and pushes right. Correct angle and angular velocity toward zero, and brake fast descent with main engine while near upright. Cut engines once both legs contact ground.",
                {"off": "Engines off", "left": "Fire left orientation engine", "main": "Fire main engine", "right": "Fire right orientation engine"})}

    def state(self):
        if self.name == "lander":
            return dict(zip(("x", "y", "vx", "vy", "angle", "angular_velocity", "left_contact", "right_contact"),
                            [round(float(x), 4) for x in self.obs]))
        e = self.env.unwrapped
        car = e.car.hull
        x, y = car.position
        index = min(range(len(e.track)), key=lambda i: (e.track[i][2]-x)**2+(e.track[i][3]-y)**2)
        bearings = []
        for ahead in (3, 8, 15):
            target = e.track[(index+ahead) % len(e.track)]
            local = car.GetLocalPoint((target[2], target[3]))
            bearings.append(round(math.degrees(math.atan2(local.x, local.y)), 1))
        return {"speed": round(car.linearVelocity.length, 2), "road_bearing_degrees_3_8_15_tiles_ahead": bearings,
                "distance_from_road_center": round(math.hypot(e.track[index][2]-x, e.track[index][3]-y), 2),
                "road_half_width": 6.67, "tiles_visited": e.tile_visited_count, "total_tiles": len(e.track)}

    def step(self, answers):
        if self.name == "racing":
            import numpy as np
            steer = {"left": -0.6, "straight": 0., "right": 0.6}[answers["steer"]["choice"]]
            pedal = answers["pedal"]["choice"]
            action = np.array([steer, .55 if pedal == "gas" else 0., .5 if pedal == "brake" else 0.], dtype=np.float32)
        else:
            action = {"off": 0, "left": 1, "main": 2, "right": 3}[answers["action"]["choice"]]
        frames = []
        for _ in range(self.repeat):
            self.obs, r, terminated, truncated, _ = self.env.step(action)
            self.last_reward = float(r)
            self.reward += float(r)
            self.tick += 1
            frames.append(Image.fromarray(self.env.render()))
            self.done = terminated or truncated
            if self.done:
                break
        return frames

    def result(self):
        result = {"reward": round(self.reward, 2), "terminated": self.done, "simulation_seconds": self.tick / self.fps}
        if self.name == "racing":
            e = self.env.unwrapped
            result.update(tiles_visited=e.tile_visited_count, total_tiles=len(e.track), success=e.tile_visited_count >= .95*len(e.track))
        else:
            result.update(success=self.done and self.last_reward == 100., final_state=self.state())
        return result

    def close(self):
        self.env.close()


class MiniGame:
    """Original turn-based demos, not implementations of commercial titles."""
    fps = 12
    limit = 80

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.tick, self.done, self.score = 0, False, 0
        self.message = "Ready"

    def scene(self, background="#142028"):
        im = Image.new("RGB", (640, 480), background)
        return im, ImageDraw.Draw(im)

    def step(self, answers):
        self.tick += 1
        self.act(answers["action"]["choice"])
        return [self.render()] * 6

    def close(self):
        pass

    def result(self):
        return {"score": self.score, "turns": self.tick, "success": self.success(), "terminated": self.done}


class Survival(MiniGame):
    title = "WILDS / Survival Craft"
    name = "survival"

    def __init__(self, seed):
        super().__init__(seed)
        self.x, self.y, self.health, self.food, self.wood = 4, 4, 10, 18, 0
        self.shelter, self.tool = False, False
        self.trees = {(1, 1), (3, 4), (6, 3), (2, 6), (7, 6), (4, 1), (6, 6)}
        self.berries = {(4, 5), (2, 2), (7, 2), (1, 5), (5, 6)}
        self.questions = {"action": choice(
            "Survive and build a shelter. Coordinates: north y-1, south y+1, east x+1, west x-1. Gather collects all wood or berries within Manhattan distance 1. Craft axe costs 2 wood and doubles wood gathering. Build shelter costs 5 wood. Eat costs 1 berry and restores 8 food. Each turn costs food, zero food costs health. Reach shelter with positive health. Choose only feasible actions; move toward resources when none adjacent.",
            {"north": "Move north", "south": "Move south", "east": "Move east", "west": "Move west", "gather": "Gather adjacent resources", "axe": "Craft axe", "shelter": "Build shelter", "eat": "Eat berry"})}
        self.inventory_berries = 0

    def state(self):
        return {"position": [self.x, self.y], "map_size": [9, 9], "health": self.health, "food": self.food,
                "wood": self.wood, "berries": self.inventory_berries, "axe": self.tool, "shelter": self.shelter,
                "trees": [list(p) for p in sorted(self.trees)], "berry_bushes": [list(p) for p in sorted(self.berries)], "last_event": self.message}

    def act(self, a):
        self.message = a
        if a in ("north", "south", "east", "west"):
            dx, dy = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}[a]
            self.x, self.y = max(0, min(8, self.x+dx)), max(0, min(8, self.y+dy))
        elif a == "gather":
            for objects, kind in ((self.trees, "wood"), (self.berries, "berries")):
                near = {p for p in objects if abs(p[0]-self.x)+abs(p[1]-self.y) <= 1}
                objects.difference_update(near)
                if kind == "wood":
                    self.wood += len(near) * (3 if self.tool else 2)
                else:
                    self.inventory_berries += 3*len(near)
        elif a == "axe" and self.wood >= 2 and not self.tool:
            self.wood -= 2
            self.tool = True
        elif a == "shelter" and self.wood >= 5:
            self.wood -= 5
            self.shelter = True
            self.score = 100
        elif a == "eat" and self.inventory_berries:
            self.inventory_berries -= 1
            self.food = min(24, self.food+8)
        else:
            self.message = "Action unavailable"
        self.food = max(0, self.food-1)
        if self.food == 0:
            self.health -= 1
        self.done = self.shelter or self.health <= 0

    def success(self):
        return self.shelter and self.health > 0

    def render(self):
        im, d = self.scene("#133f38")
        for y in range(9):
            for x in range(9):
                px, py = 104+x*46, 12+y*46
                d.rectangle((px, py, px+44, py+44), fill=("#326e50" if (x+y) % 2 else "#397657"))
                d.line((px+7, py+32, px+10, py+27), fill="#599363", width=2)
        for x, y in self.trees:
            px, py = 127+x*46, 34+y*46
            d.rectangle((px-4, py+1, px+4, py+18), fill="#b89268")
            for dx, dy in ((-9, -4), (9, -4), (0, -14)):
                d.ellipse((px+dx-13, py+dy-13, px+dx+13, py+dy+13), fill="#124b41", outline="#72ad6a", width=2)
        for x, y in self.berries:
            px, py = 127+x*46, 34+y*46
            d.ellipse((px-15, py-12, px+15, py+12), fill="#16493a")
            for dx in (-8, 0, 8):
                d.ellipse((px+dx-3, py-4, px+dx+3, py+2), fill="#f47e99")
        px, py = 127+self.x*46, 34+self.y*46
        if self.shelter:
            d.rectangle((px-21, py-12, px+21, py+20), fill="#dfcc91")
            d.polygon([(px-28, py-10), (px, py-35), (px+28, py-10)], fill="#e06857")
        else:
            d.ellipse((px-11, py-15, px+11, py+7), fill="#f1d5ad", outline="#162c37", width=2)
            d.rectangle((px-10, py+4, px+10, py+20), fill="#60c8e8")
        label(d, (20, 435), f"HP {self.health}   Food {self.food}   Wood {self.wood}   Berries {self.inventory_berries}", 19)
        return im


class Kitchen(MiniGame):
    name, title = "kitchen", "SERVICE / Kitchen Rush"
    limit = 70
    stations = {"pantry": (110, 120), "board": (300, 120), "stove": (500, 120), "pass": (500, 350), "sink": (110, 350)}

    def __init__(self, seed):
        super().__init__(seed)
        self.at, self.holding, self.pot, self.cook = "pantry", "empty", "empty", 0
        self.questions = {"action": choice(
            "Complete 3 soup orders before 70 turns. Move to a station, then interact. Pantry interaction takes raw vegetables if hands empty. Board interaction chops held raw vegetables. Stove interaction puts chopped vegetables into empty pot; soup cooks automatically in 3 turns; interact at stove with empty hands to collect ready soup (burns after 9 cooking turns). Pass interaction serves held soup. Sink interaction discards held item and clears burnt pot. Moving and interacting each cost one turn. Choose the next feasible step in this recipe.",
            {**{k: "Walk to "+k+" only. Does nothing if already there; does not use the station." for k in self.stations}, "interact": "Use current station: take ingredients, chop, load/collect pot, or serve. Does not move."})}

    def state(self):
        interaction = "unavailable"
        if self.at == "pantry" and self.holding == "empty": interaction = "Take raw vegetables"
        elif self.at == "board" and self.holding == "raw": interaction = "Chop held vegetables"
        elif self.at == "stove" and self.holding == "chopped" and self.pot == "empty": interaction = "Put vegetables into pot"
        elif self.at == "stove" and self.holding == "empty" and self.pot == "ready": interaction = "Take ready soup"
        elif self.at == "pass" and self.holding == "soup": interaction = "Serve soup"
        elif self.at == "sink": interaction = "Discard held item and clear any burnt pot"
        return {"station": self.at, "holding": self.holding, "pot": self.pot, "cooking_turns": self.cook,
                "orders_served": self.score, "orders_goal": 3, "turns_left": self.limit-self.tick, "last_event": self.message,
                "interact_action_would": interaction, "moving_to_current_station": "Does nothing; use interact to work here"}

    def act(self, a):
        self.message = a
        if a in self.stations:
            self.at = a
        elif a == "interact":
            if self.at == "pantry" and self.holding == "empty": self.holding = "raw"
            elif self.at == "board" and self.holding == "raw": self.holding = "chopped"
            elif self.at == "stove" and self.holding == "chopped" and self.pot == "empty":
                self.holding, self.pot, self.cook = "empty", "cooking", 0
            elif self.at == "stove" and self.holding == "empty" and self.pot == "ready":
                self.holding, self.pot = "soup", "empty"
            elif self.at == "pass" and self.holding == "soup":
                self.score += 1
                self.holding = "empty"
            elif self.at == "sink":
                self.holding = "empty"
                if self.pot == "burnt": self.pot = "empty"
            else: self.message = "Nothing to do here"
        if self.pot in ("cooking", "ready"):
            self.cook += 1
            self.pot = "burnt" if self.cook > 9 else "ready" if self.cook >= 3 else "cooking"
        self.done = self.score >= 3 or self.tick >= self.limit

    def success(self):
        return self.score >= 3

    def render(self):
        im, d = self.scene("#e0e9e7")
        for y in range(0, 480, 40):
            for x in range(0, 640, 40):
                if (x+y)//40 % 2: d.rectangle((x, y, x+39, y+39), fill="#c7d8d4")
        for name, (x, y) in self.stations.items():
            d.rounded_rectangle((x-64, y-48, x+64, y+48), radius=6, fill="#263e49", outline="#718b96", width=4)
            label(d, (x-50, y+20), name.upper(), 16)
            if name == "stove":
                d.ellipse((x-26, y-35, x+26, y+12), fill="#101b21", outline="#a5bcc5", width=5)
                if self.pot != "empty": d.ellipse((x-18, y-28, x+18, y+5), fill="#f2b954" if self.pot != "burnt" else "#373b40")
            elif name == "board":
                d.rectangle((x-36, y-28, x+36, y+8), fill="#e5b685")
                d.line((x-17, y-17, x+23, y-7), fill="#f0f5f7", width=6)
            elif name == "pantry":
                for offset in (-23, 0, 23): d.ellipse((x+offset-9, y-24, x+offset+9, y-6), fill="#f46d58")
            elif name == "sink":
                d.rectangle((x-35, y-32, x+35, y+8), fill="#64bada", outline="#cadce4", width=4)
            else:
                d.ellipse((x-26, y-32, x+26, y+9), fill="#fafafa")
        x, y = self.stations[self.at]
        y += 75 if y < 200 else -75
        d.ellipse((x-16, y-16, x+16, y+16), fill="#f6cb9d", outline="#243947", width=2)
        d.rectangle((x-16, y-26, x+16, y-10), fill="white")
        if self.holding != "empty": d.ellipse((x+15, y-4, x+32, y+12), fill="#f2b954")
        label(d, (210, 235), f"ORDERS  {self.score} / 3", 24, "#223d47")
        return im


class Battle(MiniGame):
    name, title = "battle", "DUEL / Element Arena"
    limit = 40

    def __init__(self, seed):
        super().__init__(seed)
        self.hp, self.enemy_hp, self.energy, self.potions = 100, 140, 3, 2
        self.intent = "attack"
        self.questions = {"action": choice(
            "Win this original turn-based robot duel. Enemy has 140 HP and player 100. Strike deals 16 damage. Ice bolt deals 34 damage against this fire enemy, costs 2 energy. Recharge gains 2 energy up to 6. Guard blocks 80% incoming damage. Heal restores 40 HP up to 100 and consumes one of two potions. Enemy attack deals 12, charged blast 30, charge deals 0 and telegraphs blast next turn. Defeat enemy before losing all HP; do not use unavailable energy/potions.",
            {"strike": "Basic strike", "ice": "Ice bolt, 2 energy", "recharge": "Recharge energy", "guard": "Guard", "heal": "Use healing potion"})}

    def state(self):
        return {"player_hp": self.hp, "enemy_hp": self.enemy_hp, "energy": self.energy, "potions": self.potions,
                "enemy_intent": self.intent, "enemy_element": "fire", "last_event": self.message}

    def act(self, a):
        damage = 0
        if a == "strike": damage = 16
        elif a == "ice" and self.energy >= 2:
            self.energy -= 2
            damage = 34
        elif a == "recharge": self.energy = min(6, self.energy+2)
        elif a == "heal" and self.potions:
            self.potions -= 1
            self.hp = min(100, self.hp+40)
        self.enemy_hp = max(0, self.enemy_hp-damage)
        incoming = {"attack": 12, "charge": 0, "blast": 30}[self.intent] if self.enemy_hp else 0
        self.hp = max(0, self.hp-int(incoming*(.2 if a == "guard" else 1)))
        self.message = f"{a.upper()}  dealt {damage} / received {int(incoming*(.2 if a == 'guard' else 1))}"
        self.intent = "blast" if self.intent == "charge" else "charge" if self.tick % 3 == 1 else "attack"
        self.done = self.hp <= 0 or self.enemy_hp <= 0
        self.score = 140-self.enemy_hp

    def success(self):
        return self.enemy_hp <= 0 and self.hp > 0

    def step(self, answers):
        super().step(answers)
        return [self.render()] * 18

    def render(self):
        im, d = self.scene("#182031")
        d.polygon([(0, 330), (640, 330), (640, 480), (0, 480)], fill="#314555")
        for x in range(-200, 900, 70): d.line((320, 260, x, 480), fill="#4d6372", width=2)
        for y in (340, 375, 425): d.line((0, y, 640, y), fill="#4d6372", width=2)
        for x, color, hp, maximum in ((160, "#6ad6ed", self.hp, 100), (480, "#ef806b", self.enemy_hp, 140)):
            d.ellipse((x-74, 334, x+74, 359), fill="#192432")
            d.rounded_rectangle((x-45, 217, x+45, 318), radius=8, fill=color, outline="#e3f0ed", width=3)
            d.rectangle((x-29, 158, x+29, 215), fill=color, outline="#e3f0ed", width=3)
            d.rectangle((x-20, 180, x+20, 192), fill="#1d2b3d")
            for side in (-1, 1):
                d.rectangle((x+side*55-9, 228, x+side*55+9, 295), fill=color)
                d.rectangle((x+side*24-10, 318, x+side*24+10, 348), fill=color)
            bar(d, x-75, 99, 150, hp/maximum, color)
            label(d, (x-58, 60), f"HP {hp}/{maximum}", 20)
        label(d, (27, 20), f"ENERGY {self.energy}   POTIONS {self.potions}", 20, "#6ad6ed")
        label(d, (355, 20), f"INTENT: {self.intent.upper()}", 20, "#ef806b")
        label(d, (25, 420), self.message, 17)
        return im


class TowerDefense(MiniGame):
    name, title = "tower", "HOLD / Tower Defense"
    limit = 80
    path = [(0, 220), (150, 220), (150, 100), (340, 100), (340, 350), (530, 350), (530, 220), (640, 220)]
    pads = [(100, 140), (235, 155), (280, 280), (420, 300), (575, 300)]

    def __init__(self, seed):
        super().__init__(seed)
        self.gold, self.lives, self.spawned = 75, 10, 0
        self.towers = [0]*5
        self.enemies = []
        self.beams = []
        self.lengths = [math.dist(a, b) for a, b in zip(self.path, self.path[1:])]
        self.questions = {"action": choice(
            "Defend against 18 incoming enemies. Build tower on empty pad for 25 gold. Upgrade existing tower for 35 gold to level 2. Towers automatically shoot nearby enemies for 12 damage per level each turn; range 155. Kills earn 12 gold. Enemies have 32-48 HP and move 32 path units/turn. One spawns every 2 turns. Paths/pads and nearby counts are supplied. Spend gold on useful coverage, upgrade crowded towers, wait when no useful purchase is affordable. Survive with lives remaining.",
            {**{f"build{i}": f"Build on pad {i}" for i in range(5)}, **{f"upgrade{i}": f"Upgrade pad {i}" for i in range(5)}, "wait": "Save gold"})}

    def point(self, distance):
        for start, end, length in zip(self.path, self.path[1:], self.lengths):
            if distance <= length:
                t = max(0, distance/length)
                return (start[0]+t*(end[0]-start[0]), start[1]+t*(end[1]-start[1]))
            distance -= length
        return self.path[-1]

    def state(self):
        return {"gold": self.gold, "lives": self.lives, "spawned": self.spawned, "total_enemies": 18,
                "pads": [{"id": i, "position": list(p), "level": self.towers[i],
                          "enemies_in_range": sum(math.dist(p, self.point(e["distance"])) <= 155 for e in self.enemies)} for i, p in enumerate(self.pads)],
                "path": [list(p) for p in self.path], "enemies": [dict(e) for e in self.enemies], "kills": self.score}

    def act(self, a):
        self.beams = []
        self.message = a
        if a.startswith("build"):
            i = int(a[-1])
            if self.gold >= 25 and not self.towers[i]:
                self.gold -= 25
                self.towers[i] = 1
            else: self.message = "Build unavailable"
        elif a.startswith("upgrade"):
            i = int(a[-1])
            if self.gold >= 35 and self.towers[i] == 1:
                self.gold -= 35
                self.towers[i] = 2
            else: self.message = "Upgrade unavailable"
        if self.tick % 2 == 1 and self.spawned < 18:
            self.enemies.append({"distance": 0, "hp": 32+(self.spawned//6)*8})
            self.spawned += 1
        for i, level in enumerate(self.towers):
            targets = [e for e in self.enemies if e["hp"] > 0 and math.dist(self.pads[i], self.point(e["distance"])) <= 155]
            if level and targets:
                target = max(targets, key=lambda e: e["distance"])
                target["hp"] -= level*12
                self.beams.append((self.pads[i], self.point(target["distance"])))
        remaining = []
        for e in self.enemies:
            if e["hp"] <= 0:
                self.gold += 12
                self.score += 1
            else:
                e["distance"] += 32
                if e["distance"] >= sum(self.lengths): self.lives -= 1
                else: remaining.append(e)
        self.enemies = remaining
        self.done = self.lives <= 0 or (self.spawned == 18 and not self.enemies)

    def success(self):
        return self.spawned == 18 and not self.enemies and self.lives > 0

    def render(self):
        im, d = self.scene("#284f49")
        for x in range(0, 640, 32): d.line((x, 0, x, 480), fill="#315a51")
        for y in range(0, 480, 32): d.line((0, y, 640, y), fill="#315a51")
        d.line(self.path, fill="#718183", width=40)
        d.line(self.path, fill="#97a2a0", width=2)
        for i, ((x, y), level) in enumerate(zip(self.pads, self.towers)):
            d.ellipse((x-25, y-25, x+25, y+25), fill="#172d38", outline="#9fb7b1", width=3)
            if level:
                d.rectangle((x-13, y-15, x+13, y+15), fill="#6bdded" if level == 1 else "#f1d773")
                d.line((x, y, x+27, y-20), fill="#e2edef", width=7)
            label(d, (x-5, y+30), i, 16)
        for e in self.enemies:
            x, y = self.point(e["distance"])
            d.rectangle((x-10, y-10, x+10, y+10), fill="#ed7782", outline="#ffd6d3", width=2)
            bar(d, x-13, y-19, 26, e["hp"]/48, "#ed7782")
        for start, end in self.beams: d.line((*start, *end), fill="#f5db83", width=3)
        label(d, (20, 430), f"LIVES {self.lives}   GOLD {self.gold}   DEFEATED {self.score}/18", 21)
        return im


def make_game(name, seed):
    if name in ("racing", "lander"):
        game = Physics(name, seed)
    else:
        game = {"survival": Survival, "tower": TowerDefense, "battle": Battle, "kitchen": Kitchen}[name](seed)
    game.seed = seed
    return game
