# Doom Demo

Jiffy plays a full Freedoom 2 level through the released `JevProtocol` adapter.
There is no Doom training, route planner, auto-aim, or scripted fallback. Four
fixed questions choose movement, turning, firing and interaction each step.
The model receives privileged engine labels, coarse depth, player variables,
and six recent positions/actions, **not screenshots**. Screenshots are recorded
for the viewer only. This is a demo, not a claim of competent level completion.
Blocked/stalled indicators are computed from depth and recent positions;
navigation instructions suggest recovery, but no code overrides model actions.

```bash
pip install -e '.[doom]'
python -m examples.doom --output artifacts/doom-demo --decisions 200
```

Requires the same supported CUDA hardware/model access as Jiffy. ViZDoom supplies
Freedoom 2; proprietary Doom assets are not required or redistributed here.
Optional `--wad`, `--map`, `--skill`, `--seed`, and `--repeat` select game settings.
Use a new output directory each time; existing recordings are never overwritten.

Outputs: `replay.gif`, `preview.png`, `decisions.jsonl` (complete model inputs,
answers and probabilities), `config.json` (contract, seed and WAD hash), and
`report.json` (kills, health, exit, reward and latency).

The engine pauses while inference runs. Replay is game-time playback, **not
real-time model performance**. Decision latency excludes one warmup call and
model loading. Reward includes a 10,000-point level-exit bonus, not an official
Doom score. Episodes are bounded by the decision budget or player death/exit.

For a CPU-only installation/recording check:

```bash
python -m examples.doom --smoke --decisions 10 --output artifacts/doom-smoke
```

Smoke uses a fixed action and is explicitly labeled; it is not a model result.
Engine documentation: https://vizdoom.farama.org/api/python/game_state/

## Initial Local Run

H100, BF16, map01, skill 3, seed 20260921, 200 decisions with eight ticks per
action: 2 kills, 8 items, 33 health remaining, no death, and no level exit.
Median decision latency was 692 ms. The episode covered 45.7 game seconds in
136.1 wall seconds including warmup but excluding model loading.

This was a development run, not a held-out benchmark: the first observation
format produced zero kills and stalled at a wall. Adding explicit blocked and
stalled indicators and recovery instructions produced the result above. No
weights were trained, and no controller overrides were added. Generalization
to other maps and seeds has not been evaluated.
