"""Run the unchanged public JevBench tasks against the local Jev protocol.

Usage: python -m benchmarks.jevbench --upstream /path/to/jevbench --output artifacts/jevbench
The upstream harness/data remain external; no private or imported tasks are invented.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.upstream.resolve()))
    from jevbench.tasks import load_jsonl, dataset_hash
    from jevbench.adapters.typesafe import TypeSafeAdapter
    from jevbench.scoring import score_task
    from jevbench.metrics import ece_top_label, latency_summary
    from jevbench.composite_v12 import intelligence, calibration, tvd, speed
    import torch
    from jiffy import DiffusionDecisions, JevProtocol
    from jiffy.diffusion_decisions import MODEL, REVISION

    tasks, tiers = [], {}
    for tier, filename in (("easy", "easy.jsonl"), ("standard", "original.jsonl"), ("hard", "hard.jsonl")):
        loaded = load_jsonl(str(args.upstream / "datasets/public" / filename))
        tasks.extend(loaded)
        tiers.update({t.id: tier for t in loaded})
    assert len(tiers) == len(tasks), "duplicate task IDs"
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "upstream_commit": subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip(),
        "task_hash": dataset_hash(tasks), "n": len(tasks), "model": MODEL, "revision": REVISION,
        "execution": "shared_document", "steps": 1, "seed": 20260921,
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((root / "jiffy").glob("*.py"))},
        "scope": "public subset only; no benchmark-specific tuning; serial local calls; no network or load adjustment in raw latency",
        "cost": None, "official_composite": None,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    torch.set_num_threads(4)
    backend = DiffusionDecisions.from_backbone()
    manifest["gpu"] = torch.cuda.get_device_name()
    api = JevProtocol(backend)
    adapter = TypeSafeAdapter(model="jiffy-diffusiongemma", key_env=None)
    # Explicit warm-up is outside measured tasks, with no benchmark labels.
    api.evaluate({"model": "jiffy-diffusiongemma", "state": "A red square.",
                  "questions": {"warmup": {"type": "noul", "instructions": "Is red mentioned?"}}})
    rows = []
    with (args.output / "predictions.jsonl").open("x") as output:
        for index, task in enumerate(tasks):
            body = adapter.build_request(task)
            started = time.perf_counter()
            try:
                result = api.evaluate(body)
                elapsed = time.perf_counter() - started
                answer = result["answers"]["decision"]
                probs = ({"yes": answer["noul"], "no": 1 - answer["noul"]}
                         if answer["type"] == "noul" else answer["probabilities"])
                scored = score_task(probs, task)
                row = {"id": task.id, "tier": tiers[task.id], "family": task.family,
                       "type": task.question["type"], "latency_s": elapsed,
                       "answer": answer, "usage": result["usage"], **scored}
            except Exception as error:
                row = {"id": task.id, "tier": tiers[task.id], "family": task.family,
                       "type": task.question["type"], "latency_s": time.perf_counter() - started,
                       "valid": False, "correct": False, "error": type(error).__name__ + ": " + str(error)}
            rows.append(row)
            output.write(json.dumps(row, allow_nan=False) + "\n")
            output.flush()
            if (index + 1) % 20 == 0 or index + 1 == len(tasks):
                print(json.dumps({"completed": index + 1, "total": len(tasks),
                                  "correct": sum(r.get("correct") is True for r in rows),
                                  "invalid": sum(not r["valid"] for r in rows)}), flush=True)

    def aggregate(selected):
        scorable = [r for r in selected if r.get("correct") is not None]
        return {"n": len(selected), "correct": sum(r["correct"] is True for r in scorable),
                "accuracy": sum(r["correct"] is True for r in scorable) / len(scorable) if scorable else None,
                "valid": sum(r["valid"] for r in selected),
                "latency": latency_summary([r["latency_s"] for r in selected])}

    by_tier = {tier: aggregate([r for r in rows if r["tier"] == tier]) for tier in ("easy", "standard", "hard")}
    by_family = {family: aggregate([r for r in rows if r["family"] == family]) for family in sorted({r["family"] for r in rows})}
    hard = [r for r in rows if r["tier"] == "hard" and r["valid"] and r.get("correct") is not None]
    ece = ece_top_label([(max(r["probs"].values()), r["correct"]) for r in hard])
    lookup = {t.id: t for t in tasks}
    tvds = [tvd(r["probs"], lookup[r["id"]].provenance["gold_probs"], lookup[r["id"]].labels)
            for r in rows if r["valid"] and lookup[r["id"]].provenance.get("gold_probs")]
    comparisons = {}
    published = json.loads((args.upstream / "results/v1.2/jevbench-v1.2-per-task.json").read_text())
    for key, system in published["systems"].items():
        matching = []
        for row in rows:
            entry = system["public_tasks"].get(row["id"])
            if entry is not None and entry[0] != "n":
                matching.append({"id": row["id"], "tier": row["tier"], "correct": entry[0] == "c"})
        comparisons[key] = {"display": system["display"], "matched_n": len(matching),
            "tiers": {tier: {"n": sum(r["tier"] == tier for r in matching),
                              "correct": sum(r["tier"] == tier and r["correct"] for r in matching)} for tier in by_tier}}
    standard_latency = by_tier["standard"]["latency"]
    report = {"manifest": manifest, "overall": aggregate(rows), "by_tier": by_tier, "by_family": by_family,
        "public_subset_intelligence": intelligence({k: v["accuracy"] for k, v in by_tier.items()}),
        "hard_public_ece": ece, "gold_distribution_n": len(tvds),
        "gold_distribution_mean_tvd": sum(tvds) / len(tvds) if tvds else None,
        "public_subset_calibration": calibration(ece["ece"], sum(tvds) / len(tvds) if tvds else None),
        "standard_only_adjusted_speed_proxy": speed(standard_latency["p50_s"], standard_latency["p95_s"], "gpu"),
        "official_speed": None, "official_cost": None, "official_composite": None,
        "comparisons_same_ids": comparisons,
        "limitations": ["Only 231/534 tasks are public: judge and private tasks absent.",
                        "Public subset Intelligence renormalizes remaining tier weights, not an official score.",
                        "Speed proxy uses standard-public only, not the full standard+judge cohort.",
                        "No hosting tariff; cost and the requested 33:0:33:33 composite remain unscored.",
                        "Local serial latency excludes the network used for hosted leaderboard rows."]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("overall", "by_tier", "public_subset_intelligence", "public_subset_calibration")}), flush=True)


if __name__ == "__main__":
    main()
