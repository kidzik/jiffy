# Tetris Placement Duel

Side-by-side Jiffy and Jev games using the existing MIT-licensed
[python-tetris engine](https://pypi.org/project/tetris/), pinned to 1.0.0a0.
The engine supplies seven-bag pieces, SRS rotation, collisions, hard drops,
line clearing, top-out and guideline scoring. Original bitmap rendering adds
the two boards, selected placements, probabilities, and decision latency.

```bash
pip install -e '.[tetris]'
python -m examples.tetris_match --pieces 60 \
  --output artifacts/tetris-jiffy-vs-jev --prompt-key
```

Uses Jiffy's normal H100 runtime and the direct TypeSafe API. The key can also
come from `TYPESAFE_API_KEY`. Credentials are not written to artifacts.
For a CPU-only engine/recording test, add `--smoke` and omit `--prompt-key`.

Both boards start empty with seed 20260921 and receive the same piece sequence.
There are no garbage attacks or holds. This is **turn-based placement selection**,
not a test of real-time keypress control. Gravity pauses for inference.

For each piece, the adapter enumerates distinct engine-reachable placements
using rotation at spawn, horizontal movement, then hard drop. It does not search
for tucks or multi-piece strategies. Both models receive the current board,
next-piece preview, and the same description of each option's immediate effects:
lines cleared, holes, stack height, surface roughness, and top-out. These are
one-piece outcome previews, not hidden model reasoning or heuristic rankings.
Legal choice sets change with the board. No heuristic selects a move; a sole
legal option is forced and explicitly logged without a model call.

The run stops at the piece budget or after both boards top out. Each side can
continue after the other loses. API versus local latency is not a kernel-speed
comparison. Replay uses 0.5 seconds per piece plus a two-second final frame,
independent of actual inference time. Score includes drop points; lines cleared
and top-out should be inspected alongside score.

Outputs: `gameplay.mp4`, `preview.png`, `final.png`, `config.json`,
`decisions.jsonl` (full requests, probabilities and chosen outcomes), and
`report.json`. One seed is a visual development demo, not a general ranking.

## First Recorded Match

Seed 20260921, 60 pieces per model, H100 BF16 Jiffy versus Jev 1.13.0:

| Model | Lines | Score | Topped Out | Median Decision |
| --- | --- | --- | --- | --- |
| Jiffy | 15 | 4,540 | No | 289 ms local |
| Jev | 17 | 4,372 | No | 152 ms including API/network |

Jev cleared more lines; Jiffy earned more engine points. Neither observation
establishes a general winner. Both used the identical one-piece-preview
interface; this is easier than playing from raw screenshots or keypresses.
