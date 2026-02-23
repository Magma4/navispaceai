# Rendering Accuracy Plan (Stairs + Doors)

Goal: prioritize *detection correctness* and topology fidelity over visual realism.

## Immediate mode (today)
- Keep legacy staircase mesh (stable placeholder).
- Focus on accurate placement/count from blueprint.
- Keep aggressive stair-zone wall carve to prevent wall/stair overlap.

## What to build next (implementation order)

### 1) Stairwell lock (deterministic)
- Build a stairwell mask from detected stair bbox + local line density.
- Remove all wall raster/vector geometry inside that mask before meshing.
- Place stairs centered in the same mask bounds.
- Acceptance: no wall strips inside stairwell in final GLB.

### 2) Door precision pipeline (2-stage)
- Stage A: high-recall proposals (current contour + edge candidates).
- Stage B: strict verifier:
  - candidate must bridge/open onto wall boundaries,
  - candidate must be near corridor-room transition,
  - reject tiny isolated shards.
- Acceptance: remove orange fragments while keeping true doors.

### 3) Detection diagnostics (already started)
- Keep writing debug overlays for each run (`/generated/debug/detect_<ts>.png`).
- Add second artifact with IDs + confidence scores for stairs/doors.
- Acceptance: every bad render can be traced to detector vs mesher quickly.

### 4) Data + training (if needed)
- Create small labeled set (20–50 plans) for stairs/doors.
- Fine-tune lightweight segmentation/detector model for blueprint symbols.
- Use confidence-calibrated post-processing from labeled validation.

## "Cracked" ideas worth trying
- Symbology-aware OCR priors: detect labels like `STAIR`, `DN`, `UP` and bias nearby region.
- Topology consistency solver: enforce door count/positions that maximize room connectivity plausibility.
- Multi-hypothesis rendering: generate top-3 stair interpretations and auto-pick by overlap score to source linework.
- Human-in-the-loop one-click correction: click true stairwell once, save anchor per blueprint family/template.

## Current status
- Legacy staircase mesh restored for stability.
- Next: implement stairwell lock + strict door verifier.
