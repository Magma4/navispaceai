import { segmentCollisionCheck } from "./CollisionManager";

/**
 * Agent animation utilities for autonomous movement, collision checks,
 * and live trajectory re-planning.
 */

/**
 * Normalize an angle delta to [-PI, PI].
 *
 * @param {number} angle
 * @returns {number}
 */
function normalizeAngle(angle) {
  const twoPi = Math.PI * 2;
  let out = angle;
  while (out > Math.PI) out -= twoPi;
  while (out < -Math.PI) out += twoPi;
  return out;
}

/**
 * Blend between two yaw angles using shortest-turn interpolation.
 *
 * @param {number} from
 * @param {number} to
 * @param {number} t
 * @returns {number}
 */
function lerpAngle(from, to, t) {
  const clamped = Math.max(0, Math.min(1, t));
  const delta = normalizeAngle(to - from);
  return from + delta * clamped;
}

/**
 * Convert backend grid path to world path.
 *
 * Supports future multi-floor points by preserving optional `y` and `floor` fields.
 *
 * @param {Array<{row:number,col:number,y?:number,floor?:string|number}>} gridPath
 * @param {number} cellSizeM
 * @returns {Array<{x:number,z:number,y:number,floor:string|number|null}>}
 */
export function gridPathToWorldPath(gridPath, cellSizeM) {
  if (!Array.isArray(gridPath) || !cellSizeM) return [];

  return gridPath.map((cell) => ({
    x: cell.col * cellSizeM,
    z: cell.row * cellSizeM,
    y: typeof cell.y === "number" ? cell.y : 0.2,
    floor: cell.floor ?? null,
  }));
}

/**
 * Build a controller that can start, stop, and re-plan path movement without remounting scene.
 *
 * @param {object} options
 * @param {Array<{x:number,z:number,y?:number,floor?:string|number|null}>} options.initialPath
 * @param {number} [options.initialSpeed=1.2]
 * @param {number[][]|null} [options.occupancyGrid=null]
 * @param {number} [options.cellSizeM=0]
 * @param {number} [options.collisionSamples=12]
 * @param {(position:{x:number,z:number,y:number}, yaw:number, meta:{segmentIndex:number,floor:string|number|null})=>void} options.onUpdate
 * @param {(info:{position:{x:number,z:number,y:number},segmentIndex:number,cell?:{row:number,col:number}})=>void} [options.onCollision]
 * @param {(info:{reason:"completed"|"stopped"|"collision"})=>void} [options.onComplete]
 * @returns {{
 *   start: () => void,
 *   stop: () => void,
 *   setPath: (nextPath:Array<{x:number,z:number,y?:number,floor?:string|number|null}>, opts?:{preserveProgress?:boolean}) => void,
 *   setSpeed: (nextSpeed:number) => void,
 *   getState: () => {running:boolean,segmentIndex:number,speed:number,pathLength:number}
 * }}
 */
export function createAgentAnimator({
  initialPath,
  initialSpeed = 1.2,
  occupancyGrid = null,
  cellSizeM = 0,
  collisionSamples = 12,
  onUpdate,
  onCollision,
  onComplete,
}) {
  let path = Array.isArray(initialPath) ? [...initialPath] : [];
  let speedMps = Math.max(0.05, Number(initialSpeed) || 1.2);
  let running = false;
  let rafId = 0;
  let segmentIndex = 0;
  let segmentT = 0;
  let previousTs = 0;

  /**
   * Safely emit final state.
   * @param {"completed"|"stopped"|"collision"} reason
   */
  function finish(reason) {
    running = false;
    cancelAnimationFrame(rafId);
    rafId = 0;
    onComplete?.({ reason });
  }

  /**
   * Tick animation frame.
   * @param {number} timestamp
   */
  function step(timestamp) {
    if (!running) return;

    if (!previousTs) previousTs = timestamp;
    const dt = Math.min(0.05, (timestamp - previousTs) / 1000);
    previousTs = timestamp;

    const a = path[segmentIndex];
    const b = path[segmentIndex + 1];

    if (!a || !b) {
      finish("completed");
      return;
    }

    const dx = b.x - a.x;
    const dz = b.z - a.z;
    const dy = (b.y ?? 0.2) - (a.y ?? 0.2);
    const segmentLength = Math.hypot(dx, dz, dy);
    const segmentDuration = Math.max(0.001, segmentLength / speedMps);

    const nextT = Math.min(1, segmentT + dt / segmentDuration);

    const from = {
      x: a.x + dx * segmentT,
      z: a.z + dz * segmentT,
      y: (a.y ?? 0.2) + dy * segmentT,
    };

    const to = {
      x: a.x + dx * nextT,
      z: a.z + dz * nextT,
      y: (a.y ?? 0.2) + dy * nextT,
    };

    // Occupancy-grid collision check along current subsegment.
    if (occupancyGrid && cellSizeM > 0) {
      const hit = segmentCollisionCheck(from, to, occupancyGrid, cellSizeM, collisionSamples);
      if (hit.hit) {
        const yaw = Math.atan2(dz, dx);
        onUpdate?.({ ...from }, yaw, { segmentIndex, floor: a.floor ?? null });
        onCollision?.({ position: { ...from }, segmentIndex, cell: hit.cell });
        finish("collision");
        return;
      }
    }

    segmentT = nextT;
    let yaw = Math.atan2(dz, dx);

    // Smooth heading transitions near corners by blending toward next segment direction.
    const c = path[segmentIndex + 2];
    if (c) {
      const nextDx = c.x - b.x;
      const nextDz = c.z - b.z;
      if (Math.hypot(nextDx, nextDz) > 1e-6) {
        const nextYaw = Math.atan2(nextDz, nextDx);
        const blendStart = 0.72;
        if (segmentT > blendStart) {
          const turnBlend = (segmentT - blendStart) / (1 - blendStart);
          yaw = lerpAngle(yaw, nextYaw, turnBlend * 0.65);
        }
      }
    }

    onUpdate?.({ ...to }, yaw, { segmentIndex, floor: a.floor ?? null });

    if (segmentT >= 1) {
      segmentIndex += 1;
      segmentT = 0;

      if (segmentIndex >= path.length - 1) {
        finish("completed");
        return;
      }
    }

    rafId = requestAnimationFrame(step);
  }

  return {
    /** Start or resume motion on current path. */
    start() {
      if (running || path.length < 2) return;
      running = true;
      previousTs = 0;
      rafId = requestAnimationFrame(step);
    },

    /** Stop motion but keep current segment progress. */
    stop() {
      if (!running) return;
      finish("stopped");
    },

    /**
     * Update trajectory in real-time (used for path re-planning while moving).
     * @param {Array<{x:number,z:number,y?:number,floor?:string|number|null}>} nextPath
     * @param {{preserveProgress?:boolean}} [opts]
     */
    setPath(nextPath, opts = {}) {
      const safePath = Array.isArray(nextPath) ? nextPath : [];
      path = safePath;

      if (!opts.preserveProgress) {
        segmentIndex = 0;
        segmentT = 0;
      } else {
        segmentIndex = Math.min(segmentIndex, Math.max(0, path.length - 2));
      }

      if (running && path.length < 2) {
        finish("stopped");
      }
    },

    /**
     * Update movement speed live.
     * @param {number} nextSpeed
     */
    setSpeed(nextSpeed) {
      speedMps = Math.max(0.05, Number(nextSpeed) || 1.2);
    },

    /**
     * Return current runtime state for UI and debugging.
     */
    getState() {
      return {
        running,
        segmentIndex,
        speed: speedMps,
        pathLength: path.length,
      };
    },
  };
}

/**
 * Backward-compatible one-shot animator helper.
 *
 * @param {object} params
 * @param {Array<{x:number,z:number,y?:number}>} params.worldPath
 * @param {number} [params.speedMps=1.2]
 * @param {(position:{x:number,z:number,y:number}, yaw:number)=>void} params.onUpdate
 * @param {() => void} [params.onComplete]
 * @returns {() => void}
 */
export function animateAgentAlongPath({ worldPath, speedMps = 1.2, onUpdate, onComplete }) {
  const animator = createAgentAnimator({
    initialPath: worldPath,
    initialSpeed: speedMps,
    onUpdate: (position, yaw) => onUpdate?.(position, yaw),
    onComplete: () => onComplete?.(),
  });

  animator.start();
  return () => animator.stop();
}
