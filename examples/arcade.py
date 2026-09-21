"""Record six actual Jiffy gameplay episodes as annotated MP4s."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from PIL import Image, ImageDraw, ImageOps

from .arcade_games import make_game, label, bar


GAMES = ("survival", "racing", "tower", "battle", "kitchen", "lander")


def overlay(scene, game, answers, step, latency, speed, smoke=False, outcome=None, provider="JIFFY / DiffusionGemma"):
    canvas = Image.new("RGB", (960, 640), "#10191e")
    d = ImageDraw.Draw(canvas)
    label(d, (20, 16), game.title, 26)
    label(d, (20, 51), "SCRIPTED SMOKE" if smoke else provider+" / typed decisions", 16, "#9eb3bd")
    fitted = ImageOps.contain(scene, (640, 480))
    canvas.paste(fitted, (16+(640-fitted.width)//2, 88+(480-fitted.height)//2))
    label(d, (685, 92), f"DECISION {step:03d}", 20, "#66dec4")
    label(d, (685, 123), f"{latency:.0f} ms {'API' if provider == 'JEV / TypeSafe' else 'inference'}", 17, "#a7bbc4")
    y = 163
    for name, answer in answers.items():
        label(d, (685, y), f"{name.upper()}: {answer['choice']}", 17)
        y += 31
        for key, probability in sorted(answer["probabilities"].items(), key=lambda item: -item[1])[:6]:
            label(d, (685, y), key, 15, "#d3dce0")
            label(d, (878, y), f"{100*probability:.0f}%", 15, "#99aeb8")
            bar(d, 685, y+22, 235, probability, "#66dec4" if key == answer["choice"] else "#4f7890")
            y += 43
        y += 12
    label(d, (20, 592), f"Structured state  |  Playback {speed:g}x  |  Seed {game.seed}", 17, "#9eb3bd")
    if outcome is not None:
        text = "OBJECTIVE COMPLETE" if outcome.get("success") else "EPISODE ENDED / objective not completed"
        d.rectangle((16, 505, 656, 568), fill="#17242e")
        label(d, (30, 518), text, 20, "#66dec4" if outcome.get("success") else "#efbb84")
    return canvas


def run(game, api, directory, args):
    import imageio_ffmpeg
    root = directory / game.name
    root.mkdir()
    provider = getattr(args, "provider", "JIFFY / DiffusionGemma")
    config = {"game": game.name, "seed": args.seed, "questions": game.questions, "engine": "Gymnasium 1.3.0 / Box2D" if game.name in ("racing", "lander") else "Original miniature game; not a commercial game",
              "policy": "scripted-smoke" if args.smoke else provider,
              "observations": "Privileged structured state, not pixels", "playback_speed": args.speed,
              "decision_limit": min(args.max_decisions, game.limit),
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(__file__).with_name("arcade_games.py"))}}
    (root / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    writer = imageio_ffmpeg.write_frames(str(root / "gameplay.mp4"), (960, 640), fps=24,
                                         codec="libx264", quality=8, pix_fmt_out="yuv420p", output_params=["-movflags", "+faststart"])
    writer.send(None)
    latencies, emitted, elapsed = [], 0, 0.
    final_frame = None
    wall_start = time.perf_counter()
    try:
        if api:
            api.evaluate({"model": "jiffy-diffusiongemma", "state": game.state(), "questions": game.questions})
        with (root / "decisions.jsonl").open("w") as log:
            for step in range(min(args.max_decisions, game.limit)):
                state = json.loads(json.dumps(game.state()))
                started = time.perf_counter()
                if api:
                    result = api.evaluate({"model": "jiffy-diffusiongemma", "state": state, "questions": game.questions})
                else:
                    answers = {}
                    for name, q in game.questions.items():
                        keys = list(q["criteria"])
                        selected = keys[step % len(keys)]
                        answers[name] = {"type": "choice", "choice": selected, "probabilities": {k: float(k == selected) for k in keys}}
                    result = {"model": "scripted-smoke", "answers": answers}
                latency = 1000*(time.perf_counter()-started)
                latencies.append(latency)
                scenes = game.step(result["answers"])
                for scene in scenes:
                    final_frame = overlay(scene, game, result["answers"], step+1, latency, args.speed, args.smoke, provider=provider)
                    elapsed += 1 / game.fps / args.speed
                    target = round(elapsed*24)
                    while emitted < target:
                        writer.send(final_frame.tobytes())
                        emitted += 1
                if step == 0 or step == min(10, game.limit//2):
                    final_frame.save(root / "preview.png")
                log.write(json.dumps({"step": step, "state": state, "result": result, "latency_ms": latency, "outcome": game.result()})+"\n")
                log.flush()
                if (step+1) % 25 == 0:
                    print(json.dumps({"game": game.name, "decisions": step+1, "latency_ms": round(latency)}), flush=True)
                if game.done:
                    break
        outcome = game.result()
        final_frame = overlay(scenes[-1], game, result["answers"], step+1, latency, args.speed, args.smoke, outcome, provider)
        final_frame.save(root / "final.png")
        for _ in range(48):
            writer.send(final_frame.tobytes())
        report = {**outcome, "game": game.name, "model": result["model"], "decisions": len(latencies),
                  "decision_p50_ms": statistics.median(latencies), "wall_seconds_including_warmup": time.perf_counter()-wall_start,
                  "video_seconds": emitted/24+2, "playback_speed": args.speed}
        (root / "report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)
        return report
    finally:
        writer.close()
        game.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", nargs="+", choices=GAMES, default=list(GAMES))
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--speed", type=float, default=2.)
    parser.add_argument("--max-decisions", type=int, default=180)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 0 < args.speed <= 8 or args.max_decisions < 1 or len(set(args.games)) != len(args.games):
        parser.error("positive decisions, unique games and playback speed in (0,8] required")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    args.output.mkdir(parents=True, exist_ok=False)
    api = None
    if not args.smoke:
        import torch
        from jiffy import DiffusionDecisions, JevProtocol
        torch.set_num_threads(4)
        api = JevProtocol(DiffusionDecisions.from_backbone())
    reports, errors = [], {}
    for name in args.games:
        try:
            reports.append(run(make_game(name, args.seed), api, args.output, args))
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
            print(json.dumps({"game": name, "error": errors[name]}), flush=True)
    (args.output / "summary.json").write_text(json.dumps({"reports": reports, "errors": errors}, indent=2)+"\n")
    if errors:
        raise SystemExit("Some games failed; see summary.json")


if __name__ == "__main__":
    main()
