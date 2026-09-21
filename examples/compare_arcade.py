"""Matched Jev gameplay and exact-state replay against saved Jiffy episodes."""
import argparse
from getpass import getpass
import json
import math
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from .arcade import GAMES, run
from .arcade_games import make_game


class Jev:
    def __init__(self, key, model="jev-latest", max_calls=1200, client=None):
        import httpx
        if not key:
            raise ValueError("Set TYPESAFE_API_KEY or use --prompt-key")
        self.client = client or httpx.Client(timeout=30, follow_redirects=False,
                                            headers={"Authorization": "Bearer "+key})
        self.model, self.calls, self.max_calls = model, 0, max_calls
        self.resolved = None

    def evaluate(self, payload):
        request = {**payload, "model": self.resolved or self.model}
        for attempt in range(3):
            if self.calls >= self.max_calls:
                raise RuntimeError("API request cap reached")
            self.calls += 1
            response = self.client.post("https://api.typesafe.ai/v1/systemone", json=request)
            if response.status_code in (429, 529) and attempt < 2:
                time.sleep(2**attempt)
                continue
            if response.status_code != 200:
                raise RuntimeError(f"TypeSafe HTTP {response.status_code}; no fallback")
            result = response.json()
            answers = result.get("answers", {})
            if set(answers) != set(payload["questions"]):
                raise ValueError("Provider returned incomplete question contract")
            for name, answer in answers.items():
                options = payload["questions"][name]["criteria"]
                probabilities = answer.get("probabilities", {})
                if answer.get("type") != "choice" or answer.get("choice") not in options or set(probabilities) != set(options):
                    raise ValueError("Provider returned invalid choice")
                p = list(probabilities.values())
                # Jev rounds each probability to two decimals; keep raw values.
                if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in p) or not math.isclose(sum(p), 1., abs_tol=.005*len(p)+1e-8):
                    raise ValueError("Provider returned invalid distribution")
            identity = result.get("model")
            if not isinstance(identity, str) or not identity:
                raise ValueError("Provider omitted model identity")
            if self.resolved and self.resolved != identity:
                raise ValueError("Provider model changed during comparison")
            self.resolved = identity
            return result
        raise RuntimeError("Retries exhausted")

    def close(self):
        self.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-key", action="store_true")
    parser.add_argument("--games", nargs="+", choices=GAMES, default=list(GAMES))
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    key = getpass("TypeSafe API key: ") if args.prompt_key else os.environ.get("TYPESAFE_API_KEY", "")
    api = Jev(key)
    del key
    args.output.mkdir(parents=True, exist_ok=False)
    comparison, errors = [], {}
    try:
        for name in args.games:
            root = args.reference / name
            config = json.loads((root / "config.json").read_text())
            source = [json.loads(line) for line in (root / "decisions.jsonl").read_text().splitlines()]
            baseline = json.loads((root / "report.json").read_text())
            game = make_game(name, config["seed"])
            game.questions = config["questions"]
            if game.state() != source[0]["state"]:
                game.close()
                raise ValueError(f"Initial state differs from reference: {name}")
            try:
                options = SimpleNamespace(seed=config["seed"], speed=config["playback_speed"],
                                          max_decisions=config["decision_limit"], smoke=False, provider="JEV / TypeSafe")
                live = run(game, api, args.output, options)
                agreements, tvds, latencies = [], [], []
                with (args.output / name / "paired.jsonl").open("w") as handle:
                    for row in source:
                        started = time.perf_counter()
                        result = api.evaluate({"model": "jiffy-diffusiongemma", "state": row["state"], "questions": config["questions"]})
                        elapsed = 1000*(time.perf_counter()-started)
                        agreement = {key: value["choice"] == row["result"]["answers"][key]["choice"] for key, value in result["answers"].items()}
                        tvd = {key: .5*sum(abs(value["probabilities"][k]/sum(value["probabilities"].values())-row["result"]["answers"][key]["probabilities"][k]/sum(row["result"]["answers"][key]["probabilities"].values())) for k in value["probabilities"]) for key, value in result["answers"].items()}
                        agreements.extend(agreement.values())
                        tvds.extend(tvd.values())
                        latencies.append(elapsed)
                        handle.write(json.dumps({"step": row["step"], "state": row["state"], "jiffy": row["result"], "jev": result,
                                                 "jev_api_ms": elapsed, "jiffy_ms": row["latency_ms"], "agreement": agreement, "tvd": tvd})+"\n")
                        handle.flush()
                item = {"game": name, "jiffy": baseline, "jev": live, "identical_initial_state": True,
                        "paired_states": len(source), "action_agreement": statistics.mean(agreements),
                        "mean_tvd": statistics.mean(tvds), "paired_jev_p50_ms": statistics.median(latencies)}
                comparison.append(item)
                print(json.dumps({"comparison": item}), flush=True)
            except Exception as error:
                errors[name] = f"{type(error).__name__}: {error}"
                print(json.dumps({"game": name, "error": errors[name]}), flush=True)
                if "HTTP 40" in str(error):
                    break
            finally:
                game.close()
            (args.output / "comparison.json").write_text(json.dumps({"games": comparison, "errors": errors, "api_attempts": api.calls}, indent=2)+"\n")
    finally:
        (args.output / "comparison.json").write_text(json.dumps({"games": comparison, "errors": errors, "api_attempts": api.calls,
            "scope": "One matched seed per game. Same saved question contracts and initial states; live trajectories diverge. Replay uses every saved Jiffy state. Agreement is not accuracy. API timings include network; Jiffy timings are local. No fallback."}, indent=2)+"\n")
        api.close()
    if errors:
        raise SystemExit("Comparison incomplete; inspect comparison.json")


if __name__ == "__main__":
    main()
