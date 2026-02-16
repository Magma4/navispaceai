/**
 * Browser-based assertion tests for frontend/scene.js.
 * Loaded by scene.test.html.
 */

import { NavScene } from "../scene.js";

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function runSceneTests() {
  const container = document.getElementById("testCanvas");
  const navScene = new NavScene(container);

  navScene.ensureAgent();
  assert(navScene.agent !== null, "Agent should be created");

  navScene.updateAgent({ x: 1.0, z: 2.0 }, Math.PI / 4);
  assert(Math.abs(navScene.agent.position.x - 1.0) < 1e-6, "Agent X position should update");
  assert(Math.abs(navScene.agent.position.z - 2.0) < 1e-6, "Agent Z position should update");

  navScene.setClickMode(true, () => {});
  assert(navScene.clickEnabled === true, "Click mode should enable");

  navScene.setFollowAgent(true);
  assert(navScene.followAgent === true, "Follow mode should enable");

  navScene.setPath([
    { x: 0, z: 0 },
    { x: 1, z: 1 },
  ]);
  assert(navScene.pathLine !== null, "Path line should be created for 2+ points");

  navScene.setPath([{ x: 0, z: 0 }]);
  assert(navScene.pathLine === null, "Path line should be removed for insufficient points");

  const status = document.getElementById("status");
  status.textContent = "scene.test.js: all tests passed";
}

window.addEventListener("DOMContentLoaded", () => {
  try {
    runSceneTests();
  } catch (err) {
    const status = document.getElementById("status");
    status.textContent = `scene.test.js: FAILED - ${err.message}`;
    throw err;
  }
});
