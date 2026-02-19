# NavispaceAI Project Board

## Features
- [x] Blueprint preprocessing pipeline
- [x] Wall and door detection
- [x] Occupancy grid generation and obstacle inflation
- [x] A* pathfinding backend endpoint
- [x] 3D model generation and GLB export
- [x] Three.js autonomous navigation mode
- [ ] Multi-floor support
- [ ] Room labeling and semantic metadata
- [ ] Dynamic re-routing during agent movement

## Bugs
- [ ] Door heuristic can over-detect on noisy scans
- [ ] Some blueprints produce disconnected occupancy islands due to aggressive inflation
- [ ] Start/goal selection has no explicit occupied-cell warning in UI
- [ ] Scene click picking assumes horizontal floor plane only

## Enhancements
- [ ] Add path smoothing (Bezier/Catmull-Rom) with obstacle checks
- [ ] Add backend cache for repeated blueprint uploads
- [ ] Add API versioning and schema examples
- [ ] Add e2e browser automation tests for full navigation loop
- [ ] Add CI pipeline for lint + unit + integration tests
