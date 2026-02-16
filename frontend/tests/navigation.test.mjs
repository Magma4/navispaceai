/**
 * Assertion-based unit tests for frontend/navigation.js.
 * Run with: node frontend/tests/navigation.test.mjs
 */

import assert from "node:assert/strict";
import { AgentNavigator } from "../navigation.js";

function makeMockScene() {
  return {
    ensured: false,
    updates: [],
    ensureAgent() {
      this.ensured = true;
    },
    updateAgent(pos, yaw) {
      this.updates.push({ pos, yaw });
    },
  };
}

async function testFetchPathSuccess() {
  const scene = makeMockScene();
  const nav = new AgentNavigator(scene);

  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ path: [{ row: 0, col: 0 }], world_path: [{ x: 0, z: 0 }] }),
  });

  const result = await nav.fetchPath("http://localhost:8000", { row: 0, col: 0 }, { row: 1, col: 1 });
  assert.equal(Array.isArray(result.path), true);
}

async function testFetchPathFailure() {
  const scene = makeMockScene();
  const nav = new AgentNavigator(scene);

  globalThis.fetch = async () => ({
    ok: false,
    status: 400,
    json: async () => ({ detail: "Invalid path query" }),
  });

  await assert.rejects(
    () => nav.fetchPath("http://localhost:8000", { row: 0, col: 0 }, { row: 1, col: 1 }),
    /Invalid path query/
  );
}

function testAnimateAlongPath() {
  const scene = makeMockScene();
  const nav = new AgentNavigator(scene);

  let now = 0;
  globalThis.performance = { now: () => now };

  let queue = [];
  globalThis.requestAnimationFrame = (cb) => {
    queue.push(cb);
    return queue.length;
  };
  globalThis.cancelAnimationFrame = () => {};

  nav.animateAlongPath(
    [
      { x: 0, z: 0 },
      { x: 1, z: 0 },
      { x: 2, z: 0 },
    ],
    2.0
  );

  for (let i = 0; i < 50 && queue.length > 0; i += 1) {
    const cb = queue.shift();
    now += 20;
    cb(now);
  }

  assert.equal(scene.ensured, true);
  assert.ok(scene.updates.length > 0);
  const last = scene.updates[scene.updates.length - 1];
  assert.ok(last.pos.x >= 1.8);
}

async function run() {
  await testFetchPathSuccess();
  await testFetchPathFailure();
  testAnimateAlongPath();
  console.log("navigation.test.mjs: all tests passed");
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
