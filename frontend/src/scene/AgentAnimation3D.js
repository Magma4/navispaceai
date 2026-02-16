/**
 * AgentAnimation3D provides path-follow animation in world meters (x, y, z)
 * with smooth interpolation, yaw alignment, and multi-floor continuity.
 */

/**
 * Create a re-usable 3D animator controller.
 *
 * @param {object} options
 * @param {Array<{x:number,y:number,z:number,floor?:number}>} options.initialPath
 * @param {number} [options.initialSpeed=1.2]
 * @param {(position:{x:number,y:number,z:number}, yaw:number, meta:{segmentIndex:number,floor:number|null})=>void} options.onUpdate
 * @param {(info:{reason:"completed"|"stopped"})=>void} [options.onComplete]
 * @returns {{
 *   start:()=>void,
 *   stop:()=>void,
 *   setPath:(path:Array<{x:number,y:number,z:number,floor?:number}>)=>void,
 *   setSpeed:(speed:number)=>void,
 *   isRunning:()=>boolean
 * }}
 */
export function createAgentAnimator3D({
  initialPath,
  initialSpeed = 1.2,
  onUpdate,
  onComplete,
}) {
  let path = Array.isArray(initialPath) ? [...initialPath] : [];
  let speed = Math.max(0.05, Number(initialSpeed) || 1.2);
  let running = false;
  let rafId = 0;
  let seg = 0;
  let t = 0;
  let prevTs = 0;

  /** Finish current animation run. */
  function finish(reason) {
    running = false;
    cancelAnimationFrame(rafId);
    rafId = 0;
    onComplete?.({ reason });
  }

  /** Frame step for interpolation along current path segment. */
  function frame(ts) {
    if (!running) return;
    if (!prevTs) prevTs = ts;

    const dt = Math.min(0.05, (ts - prevTs) / 1000);
    prevTs = ts;

    const a = path[seg];
    const b = path[seg + 1];
    if (!a || !b) {
      finish("completed");
      return;
    }

    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dz = b.z - a.z;

    const length = Math.hypot(dx, dy, dz);
    const duration = Math.max(0.001, length / speed);

    t = Math.min(1, t + dt / duration);

    const pos = {
      x: a.x + dx * t,
      y: a.y + dy * t,
      z: a.z + dz * t,
    };

    const yaw = Math.atan2(dz, dx);
    onUpdate?.(pos, yaw, { segmentIndex: seg, floor: b.floor ?? null });

    if (t >= 1) {
      seg += 1;
      t = 0;
      if (seg >= path.length - 1) {
        finish("completed");
        return;
      }
    }

    rafId = requestAnimationFrame(frame);
  }

  return {
    start() {
      if (running || path.length < 2) return;
      running = true;
      prevTs = 0;
      rafId = requestAnimationFrame(frame);
    },

    stop() {
      if (!running) return;
      finish("stopped");
    },

    setPath(nextPath) {
      path = Array.isArray(nextPath) ? [...nextPath] : [];
      seg = 0;
      t = 0;
      if (running && path.length < 2) {
        finish("stopped");
      }
    },

    setSpeed(nextSpeed) {
      speed = Math.max(0.05, Number(nextSpeed) || 1.2);
    },

    isRunning() {
      return running;
    },
  };
}
