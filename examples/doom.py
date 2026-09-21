"""Full-level Freedoom demo using Jiffy's typed decisions, not a scripted policy."""
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


QUESTIONS = {
    "movement": {
        "type": "choice",
        "instructions": "Choose movement to explore the level, survive, collect supplies and reach the exit. When navigation.blocked_ahead or navigation.stalled is true, back away to make room for turning; do not move forward into a wall. Otherwise advance into open space.",
        "criteria": {"forward": "Advance", "back": "Back away", "left": "Strafe left", "right": "Strafe right", "stay": "Stand still"},
    },
    "turn": {
        "type": "choice",
        "instructions": "Choose a turn. When navigation.blocked_ahead or navigation.stalled is true, turn toward the side with more clearance (left on a tie). Otherwise aim at visible living enemies, or keep heading into open space. Screen x increases to the right.",
        "criteria": {"left": "Turn left", "right": "Turn right", "none": "Keep heading"},
    },
    "fire": {"type": "noul", "instructions": "Should we shoot now? Fire when a visible living enemy is near the center of the screen and ammunition is available. Do not shoot corpses or walls."},
    "use": {"type": "noul", "instructions": "Should we interact now to open a nearby door or activate a switch? Try when facing a nearby obstruction while exploring."},
}
BUTTONS = ["MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT", "TURN_LEFT", "TURN_RIGHT", "ATTACK", "USE"]
VARIABLES = ["HEALTH", "ARMOR", "SELECTED_WEAPON_AMMO", "KILLCOUNT", "ITEMCOUNT", "SECRETCOUNT", "POSITION_X", "POSITION_Y", "ANGLE"]


def command(answers):
    move, turn = answers["movement"]["choice"], answers["turn"]["choice"]
    if move not in QUESTIONS["movement"]["criteria"] or turn not in QUESTIONS["turn"]["criteria"]:
        raise ValueError("invalid action")
    fire, use = answers["fire"]["noul"], answers["use"]["noul"]
    if not all(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1 for p in (fire, use)):
        raise ValueError("invalid action probability")
    return [move == key for key in ("forward", "back", "left", "right")] + [turn == "left", turn == "right", fire >= .5, use >= .5]


def observe(game, vzd, history):
    frame = game.get_state()
    stats = {key.lower(): round(game.get_game_variable(getattr(vzd.GameVariable, key)), 2) for key in VARIABLES}
    objects = [{"name": label.object_name, "screen_x": round((label.x + label.width / 2) / 320, 3),
                "screen_width": round(label.width / 320, 3),
                "distance": round(math.hypot(label.object_position_x - stats["position_x"], label.object_position_y - stats["position_y"]))}
               for label in frame.labels if label.object_name != "DoomPlayer"]
    depth = frame.depth_buffer
    clearance = {name: round(float(depth[90:150, start:end].mean()), 1)
                 for name, start, end in (("left", 0, 100), ("front", 110, 210), ("right", 220, 320))}
    stalled = len(history) >= 3 and all(math.hypot(stats["position_x"] - h["x"], stats["position_y"] - h["y"]) < 8 for h in list(history)[-3:])
    return {"objective": "Explore, survive, find and activate the level exit. Doors may need use. This is a full level, not a shooting arena.",
            "player": stats, "visible_objects": sorted(objects, key=lambda x: x["distance"])[:24],
            "navigation": {"blocked_ahead": clearance["front"] < 12, "stalled": stalled},
            "clearance": clearance,
            "clearance_units": "Relative depth 0-255; larger is farther/open. Not world distance.",
            "recent_steps": list(history)}


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wad", type=Path)
    parser.add_argument("--map", default="map01")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--skill", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--decisions", type=positive, default=200)
    parser.add_argument("--repeat", type=positive, default=8)
    parser.add_argument("--smoke", action="store_true", help="Scripted engine check only; does not load Jiffy")
    args = parser.parse_args()
    import vizdoom as vzd
    from PIL import Image, ImageDraw

    wad = args.wad or Path(vzd.__file__).parent / "freedoom2.wad"
    if not wad.is_file():
        parser.error("WAD not found; supply --wad /path/to/freedoom2.wad")
    args.output.mkdir(parents=True, exist_ok=False)
    config = {"map": args.map, "seed": args.seed, "skill": args.skill, "decisions": args.decisions,
              "repeat": args.repeat, "wad_sha256": hashlib.sha256(wad.read_bytes()).hexdigest(),
              "questions": QUESTIONS, "buttons": BUTTONS, "policy": "scripted-smoke" if args.smoke else "jiffy",
              "observation": "Privileged engine labels, depth and player variables; not pixel-only.",
              "timing": "Synchronous game: paused during inference. Replay uses game time, not wall time."}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    api = None
    if not args.smoke:
        import torch
        from jiffy import DiffusionDecisions, JevProtocol
        torch.set_num_threads(4)
        api = JevProtocol(DiffusionDecisions.from_backbone())
    game = vzd.DoomGame()
    frames, durations, latencies = [], [], []
    history = deque(maxlen=6)
    reward = 0.
    start = time.perf_counter()
    try:
        game.set_doom_config_path(str(args.output / "engine.ini"))
        game.set_doom_game_path(str(wad))
        game.set_doom_map(args.map)
        game.set_doom_skill(args.skill)
        game.set_seed(args.seed)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_labels_buffer_enabled(True)
        game.set_depth_buffer_enabled(True)
        game.set_available_buttons([getattr(vzd.Button, name) for name in BUTTONS])
        game.set_episode_timeout(args.decisions * args.repeat + 1)
        game.set_map_exit_reward(10000.)
        game.init()
        game.new_episode()
        if api:
            api.evaluate({"model": "jiffy-diffusiongemma", "state": observe(game, vzd, history), "questions": QUESTIONS})
        with (args.output / "decisions.jsonl").open("w") as log:
            for step in range(args.decisions):
                if game.is_episode_finished():
                    break
                state = observe(game, vzd, history)
                began = time.perf_counter()
                if api:
                    result = api.evaluate({"model": "jiffy-diffusiongemma", "state": state, "questions": QUESTIONS})
                    answers = result["answers"]
                else:
                    answers = {"movement": {"choice": "forward"}, "turn": {"choice": "left"}, "fire": {"noul": 0.}, "use": {"noul": 0.}}
                    result = {"model": "scripted-smoke", "answers": answers}
                elapsed = 1000 * (time.perf_counter() - began)
                latencies.append(elapsed)
                action = command(answers)
                tick_start = game.get_episode_time()
                for tick in range(args.repeat):
                    if game.is_episode_finished():
                        break
                    if tick % 4 == 0:
                        frame = Image.new("RGB", (320, 292), "#14181a")
                        frame.paste(Image.fromarray(game.get_state().screen_buffer.copy()))
                        draw = ImageDraw.Draw(frame)
                        draw.text((6, 243), f"{'SMOKE' if args.smoke else 'JIFFY'} | {args.map} | decision {step + 1} | {elapsed:.0f} ms", fill="white")
                        draw.text((6, 258), f"{answers['movement']['choice']} / {answers['turn']['choice']} | fire {int(action[6])} use {int(action[7])}", fill="#80dfbd")
                        draw.text((6, 273), "Structured state | replay: game time", fill="#aab4bd")
                        frames.append(frame)
                        durations.append(0)
                    # Release USE between decisions so successive interactions register.
                    buttons = action.copy()
                    buttons[7] = action[7] and tick == 0
                    reward += game.make_action(buttons, 1)
                    durations[-1] += 1000 / 35
                history.append({"x": state["player"]["position_x"], "y": state["player"]["position_y"],
                                "angle": state["player"]["angle"], "movement": answers["movement"]["choice"], "turn": answers["turn"]["choice"]})
                log.write(json.dumps({"step": step, "state": state, "result": result, "decision_ms": elapsed,
                                      "ticks": game.get_episode_time() - tick_start, "reward_total": reward}) + "\n")
                log.flush()
                if (step + 1) % 25 == 0:
                    print(json.dumps({"decisions": step + 1, "reward": reward, "decision_ms": round(elapsed)}), flush=True)
        report = {"policy": config["policy"], "decisions": len(latencies), "reward": reward,
                  "exit": reward >= 10000, "dead": game.is_player_dead(),
                  "kills": game.get_game_variable(vzd.GameVariable.KILLCOUNT),
                  "health": game.get_game_variable(vzd.GameVariable.HEALTH),
                  "game_seconds": game.get_episode_time() / 35,
                  "wall_seconds_including_warmup": time.perf_counter() - start,
                  "decision_p50_ms": statistics.median(latencies) if latencies else None}
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        game.close()
        if frames:
            frames[0].save(args.output / "replay.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0)
            frames[len(frames) // 2].save(args.output / "preview.png")


if __name__ == "__main__":
    main()
