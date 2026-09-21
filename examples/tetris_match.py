"""Turn-based Tetris placement duel using the python-tetris rules engine."""
import argparse
import copy
from getpass import getpass
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import tetris
from tetris.engine import Gravity
from tetris.impl.queue import SevenBag
from tetris.impl.rotation import SRS
from tetris.impl.scorer import GuidelineScorer
from PIL import Image, ImageDraw

from .arcade_games import label


class TurnGravity(Gravity):
    def calculate(self, delta=None):
        """The model chooses one placement per turn; wall-clock gravity is off."""


def new_game(seed):
    return tetris.BaseGame((TurnGravity, SevenBag, SRS, GuidelineScorer), seed=seed)


def features(board):
    heights, holes = [], 0
    for col in range(board.shape[1]):
        occupied = [r for r in range(board.shape[0]) if board[r, col]]
        heights.append(board.shape[0]-occupied[0] if occupied else 0)
        if occupied:
            holes += sum(not bool(board[r, col]) for r in range(occupied[0], board.shape[0]))
    return {"holes": holes, "max_height": max(heights), "aggregate_height": sum(heights),
            "roughness": sum(abs(a-b) for a, b in zip(heights, heights[1:])), "column_heights": heights}


def placements(game):
    """Enumerate engine-reachable rotate-then-slide-then-drop placements."""
    found, seen = {}, set()
    for turns in range(4):
        for column in range(game.width):
            # Ruleset's attribute proxy cannot be deep-copied; moves only read it.
            candidate = copy.deepcopy(game, {id(game.rules): game.rules, id(game.board): game.board.copy()})
            if turns:
                candidate.rotate(turns)
            left = min(y+candidate.piece.y for _, y in candidate.piece.minos)
            candidate.drag(column-left)
            if min(y+candidate.piece.y for _, y in candidate.piece.minos) != column:
                continue
            pose = copy.deepcopy(candidate.piece)
            candidate.hard_drop()
            signature = (candidate.board.tobytes(), candidate.lost)
            if signature in seen:
                continue
            seen.add(signature)
            key = f"r{turns}c{column}"
            found[key] = {"game": candidate, "pose": pose, "drop": candidate.delta.x,
                          "facts": {"clockwise_quarter_turns": turns, "left_column": column,
                                    "lines_cleared": len(candidate.delta.clears), "tops_out": candidate.lost,
                                    **features(candidate.board[-game.height:])}}
    return found


def request(game, choices):
    state = {"board_top_to_bottom": ["".join("#" if cell else "." for cell in row) for row in game.board[-game.height:]],
             "piece": game.piece.type.name, "next": [p.name for p in list(game.queue)[:4]],
             "current_board": features(game.board[-game.height:])}
    instructions = ("Choose the best legal Tetris placement to clear lines and survive. Each option contains exact "
                    "one-piece lookahead from the game engine, not a recommendation. Avoid topping out and creating "
                    "holes (empty cells with blocks above them). Prefer clearing lines, a low stack and an even surface. "
                    "Column 0 is leftmost. Evaluate the stated resulting-board metrics; do not simply choose the first option.")
    return {"model": "jiffy-diffusiongemma", "state": state, "questions": {"placement": {
        "type": "choice", "instructions": instructions,
        "criteria": {key: value["facts"] for key, value in choices.items()}}}}


COLORS = {0: "#121e24", 1: "#62d3e9", 2: "#568be8", 3: "#f3a456", 4: "#eddb70", 5: "#69d39c", 6: "#bd8ce8", 7: "#f1768b"}


def render(games, lines, counts, answers, latencies, animation=None, progress=1., smoke=False, final=False):
    im = Image.new("RGB", (1024, 768), "#10191e")
    d = ImageDraw.Draw(im)
    label(d, (40, 20), "BLOCK / TETRIS DUEL", 28)
    label(d, (40, 58), "SCRIPTED SMOKE" if smoke else "Jiffy vs Jev / same piece sequence", 17, "#9cb5bd")
    for i, (game, x, name, accent) in enumerate(zip(games, (80, 704), ("JIFFY", "JEV"), ("#69d9c3", "#f6bb73"))):
        label(d, (x, 99), name, 23, accent)
        board = game.board[-game.height:].copy()
        if animation and animation[i]:
            old_board, pose, drop = animation[i]
            board = old_board[-game.height:].copy()
            row = pose.x + round(drop*progress)-game.height
            for dr, dc in pose.minos:
                r, c = row+dr, pose.y+dc
                if 0 <= r < game.height and 0 <= c < game.width:
                    board[r, c] = pose.type
        for r, row in enumerate(board):
            for c, value in enumerate(row):
                px, py = x+c*24, 143+r*24
                d.rectangle((px, py, px+22, py+22), fill=COLORS.get(int(value), "#a5b5ba"))
                if value:
                    d.line((px+2, py+2, px+20, py+2), fill="#e5eef0", width=2)
        d.rectangle((x-2, 141, x+240, 624), outline="#47606a", width=2)
        label(d, (x, 645), f"{lines[i]} LINES   {game.score} PTS", 20, accent)
        label(d, (x, 680), f"{counts[i]} pieces / {'TOP OUT' if game.lost else 'playing'}", 17)
        label(d, (x, 711), f"{latencies[i]:.0f} ms {'local' if i == 0 else 'API'}", 16, "#9cb5bd")
    label(d, (399, 153), "PLACEMENT", 20, "#9cb5bd")
    for i, y in enumerate((205, 390)):
        answer = answers[i]
        label(d, (391, y), "JIFFY" if i == 0 else "JEV", 20, "#69d9c3" if i == 0 else "#f6bb73")
        if answer:
            label(d, (391, y+34), answer["choice"], 25)
            for j, (key, p) in enumerate(sorted(answer["probabilities"].items(), key=lambda item: -item[1])[:3]):
                label(d, (391, y+77+j*24), f"{key}   {100*p:.0f}%", 16, "#b7c9cf")
    label(d, (384, 611), "TURN-BASED", 18)
    label(d, (371, 643), "One-ply state preview", 16, "#9cb5bd")
    label(d, (379, 670), "Replay / not real time", 15, "#9cb5bd")
    if final:
        label(d, (389, 718), "RUN COMPLETE", 19, "#eddb70")
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pieces", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--prompt-key", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.pieces < 1 or args.pieces > 300:
        parser.error("pieces must be in 1..300")
    import imageio_ffmpeg
    apis = [None, None]
    args.output.mkdir(parents=True, exist_ok=False)
    games = [new_game(args.seed), new_game(args.seed)]
    lines, counts, timings = [0, 0], [0, 0], [[], []]
    answers, latencies = [None, None], [0., 0.]
    config = {"seed": args.seed, "piece_limit": args.pieces, "engine": "tetris==1.0.0a0, SevenBag, SRS, GuidelineScorer",
              "scope": "Turn-based placement selection. Same seeded bags, independent boards, no garbage attacks or hold. Both models get unranked legal one-piece outcome metrics. No policy fallback. Replay 0.5 seconds per piece, not wall time.",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "smoke": args.smoke}
    (args.output / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    writer = None
    try:
        if not args.smoke:
            from .compare_arcade import Jev
            key = getpass("TypeSafe API key: ") if args.prompt_key else os.environ.get("TYPESAFE_API_KEY", "")
            apis[1] = Jev(key, max_calls=args.pieces+6)
            del key
            import torch
            from jiffy import DiffusionDecisions, JevProtocol
            torch.set_num_threads(4)
            apis[0] = JevProtocol(DiffusionDecisions.from_backbone())
            warmup = request(games[0], placements(games[0]))
            for api in apis:
                api.evaluate(warmup)
        writer = imageio_ffmpeg.write_frames(str(args.output / "gameplay.mp4"), (1024, 768), fps=24,
                                             codec="libx264", quality=8, output_params=["-movflags", "+faststart"])
        writer.send(None)
        with (args.output / "decisions.jsonl").open("w") as log:
            for turn in range(args.pieces):
                animation = [None, None]
                for player in (0, 1):
                    game = games[player]
                    if game.lost:
                        continue
                    choices = placements(game)
                    payload = request(game, choices)
                    started = time.perf_counter()
                    if len(choices) == 1:
                        key = next(iter(choices))
                        result = {"model": "forced-only-legal-placement", "answers": {"placement": {"choice": key, "probabilities": {key: 1.}}}}
                    elif args.smoke:
                        key = list(choices)[turn % len(choices)]
                        result = {"model": "scripted-smoke", "answers": {"placement": {"choice": key, "probabilities": {k: float(k == key) for k in choices}}}}
                    else:
                        result = apis[player].evaluate(payload)
                    latencies[player] = 1000*(time.perf_counter()-started)
                    timings[player].append(latencies[player])
                    answers[player] = result["answers"]["placement"]
                    selected = choices[answers[player]["choice"]]
                    animation[player] = (game.board.copy(), selected["pose"], selected["drop"])
                    games[player] = selected["game"]
                    lines[player] += selected["facts"]["lines_cleared"]
                    counts[player] += 1
                    log.write(json.dumps({"turn": turn, "player": "jiffy" if player == 0 else "jev", "request": payload,
                                          "result": result, "decision_ms": latencies[player], "chosen_outcome": selected["facts"],
                                          "lines_total": lines[player], "score": games[player].score})+"\n")
                    log.flush()
                for frame in range(12):
                    im = render(games, lines, counts, answers, latencies, animation if frame < 9 else None,
                                min(1., frame/8), args.smoke)
                    writer.send(im.tobytes())
                if turn == 0 or turn == 10:
                    im.save(args.output / "preview.png")
                if (turn+1) % 10 == 0:
                    print(json.dumps({"pieces": counts, "lines": lines, "scores": [g.score for g in games]}), flush=True)
                if all(g.lost for g in games):
                    break
        final = render(games, lines, counts, answers, latencies, smoke=args.smoke, final=True)
        final.save(args.output / "final.png")
        for _ in range(48):
            writer.send(final.tobytes())
        report = {name: {"pieces": counts[i], "lines": lines[i], "score": games[i].score, "top_out": games[i].lost,
                         "decision_p50_ms": statistics.median(timings[i])} for i, name in enumerate(("jiffy", "jev"))}
        report["scope"] = config["scope"]
        (args.output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)
    finally:
        if writer:
            writer.close()
        if apis[1]:
            apis[1].close()


if __name__ == "__main__":
    main()
