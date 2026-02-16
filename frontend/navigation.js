/**
 * Navigation orchestration: backend path request + smooth agent animation.
 */

export class AgentNavigator {
  /**
   * @param {import('./scene.js').NavScene} navScene
   */
  constructor(navScene) {
    this.navScene = navScene;
    this.animationFrame = null;
  }

  /**
   * Request a path from backend A* service.
   * @param {string} baseUrl
   * @param {{row:number,col:number}} start
   * @param {{row:number,col:number}} goal
   */
  async fetchPath(baseUrl, start, goal) {
    const res = await fetch(`${baseUrl}/find-path`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start, goal, diagonal: true }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Path request failed with ${res.status}`);
    }

    return res.json();
  }

  /**
   * Animate the agent along world-space points.
   * @param {Array<{x:number,z:number}>} worldPath
   * @param {number} speedMps
   */
  animateAlongPath(worldPath, speedMps = 1.2) {
    if (!worldPath || worldPath.length < 2) return;

    this.navScene.ensureAgent();
    if (this.animationFrame) cancelAnimationFrame(this.animationFrame);

    const points = worldPath;
    let segIdx = 0;
    let t = 0;
    let prevTs = performance.now();

    const step = (ts) => {
      const dt = Math.min(0.05, (ts - prevTs) / 1000);
      prevTs = ts;

      const a = points[segIdx];
      const b = points[segIdx + 1];
      if (!a || !b) return;

      const dx = b.x - a.x;
      const dz = b.z - a.z;
      const segLen = Math.hypot(dx, dz);
      const segDur = Math.max(0.001, segLen / speedMps);

      t += dt / segDur;
      if (t >= 1) {
        segIdx += 1;
        t = 0;
        if (segIdx >= points.length - 1) {
          this.navScene.updateAgent(points[points.length - 1], Math.atan2(dz, dx));
          return;
        }
      }

      const currA = points[segIdx];
      const currB = points[segIdx + 1];
      const ix = currA.x + (currB.x - currA.x) * t;
      const iz = currA.z + (currB.z - currA.z) * t;
      const yaw = Math.atan2(currB.z - currA.z, currB.x - currA.x);

      this.navScene.updateAgent({ x: ix, z: iz }, yaw);
      this.animationFrame = requestAnimationFrame(step);
    };

    this.animationFrame = requestAnimationFrame(step);
  }
}
