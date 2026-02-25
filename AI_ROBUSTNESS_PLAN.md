# Navispace AI Integration + Robustness Plan

## Product Goal
Turn blueprint-to-navigation into a reliable system that improves from real user corrections over time.

## Phase 1 (Now) — Stabilize + Observe
- ✅ Keep simple stair mesh as default for rendering stability.
- ✅ Strict detection mode (`strict_mode`) for conservative production runs.
- ✅ Debug bundle outputs (`/generated/debug/*`) for fast failure diagnosis.
- ✅ Annotation feedback endpoint (`POST /feedback/annotations`) to capture corrected walls/doors/stairs.

## Phase 2 (Next 1-2 sprints) — Learning Loop
1. Build lightweight correction UI:
   - draw/edit boxes for doors/stairs
   - line adjust for walls
   - submit to `/feedback/annotations`
2. Dataset builder job:
   - transform feedback JSONL into train/val set
   - merge with existing labeled data
3. Weekly retrain job (`backend/ml/train.py`) with model versioning
4. Shadow deployment:
   - run ML + heuristic in parallel
   - compare IoU/F1 and route quality before promoting

## Phase 3 — Hybrid Intelligence
- Replace pure thresholding with class confidence maps.
- Use ML masks to seed geometry + heuristics to enforce topology constraints.
- Add uncertainty scoring and human-review queue for low-confidence outputs.

## Core Metrics (must track)
- Wall IoU / F1
- Door precision-recall
- Stair false-positive rate
- Path success rate (start→goal)
- Rendering defect rate (floating fragments / broken walls)
- Time-to-correct (human effort)

## Engineering Hardening
- Keep strict/non-strict A/B mode in API.
- Golden blueprint regression suite (10-20 real plans).
- Auto-fail CI if path success or geometry metrics regress.
- Add model/version metadata to process response for traceability.

## Immediate Build Order
1. Frontend strict-mode checkbox + feedback submit action.
2. Offline script: convert feedback JSONL into training tensors/masks.
3. Evaluation harness to score old vs new checkpoints on golden set.
4. CI gate for "no quality regression".
