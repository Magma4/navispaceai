/**
 * Frontend app bootstrap and interaction logic.
 */

import { NavScene } from "./scene.js";
import { AgentNavigator } from "./navigation.js";

const BACKEND_BASE = "http://localhost:8000";

const els = {
  blueprintInput: document.getElementById("blueprintInput"),
  processBtn: document.getElementById("processBtn"),
  navModeBtn: document.getElementById("navModeBtn"),
  followCamera: document.getElementById("followCamera"),
  status: document.getElementById("status"),
  sceneContainer: document.getElementById("sceneContainer"),
};

const scene = new NavScene(els.sceneContainer);
const navigator = new AgentNavigator(scene);

let latestMeta = null;
let navMode = false;
let clickPoints = [];

/**
 * Set status text for user feedback.
 * @param {string} text
 */
function setStatus(text) {
  els.status.textContent = text;
}

/**
 * Convert clicked world coordinate to backend grid point.
 * @param {{x:number,z:number}} world
 */
function worldToGrid(world) {
  const cell = latestMeta.cell_size_m;
  const rows = latestMeta.grid_shape.rows;
  const cols = latestMeta.grid_shape.cols;

  const row = Math.max(0, Math.min(rows - 1, Math.round(world.z / cell)));
  const col = Math.max(0, Math.min(cols - 1, Math.round(world.x / cell)));
  return { row, col };
}

/** Process the selected blueprint image on backend. */
async function processBlueprint() {
  const file = els.blueprintInput.files?.[0];
  if (!file) {
    setStatus("Select a blueprint image first.");
    return;
  }

  const fd = new FormData();
  fd.append("file", file);

  setStatus("Processing blueprint...");
  els.processBtn.disabled = true;

  try {
    const res = await fetch(`${BACKEND_BASE}/process-blueprint`, {
      method: "POST",
      body: fd,
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Processing failed (${res.status})`);
    }

    latestMeta = await res.json();
    await scene.loadModel(`${BACKEND_BASE}${latestMeta.model_url}`);

    els.navModeBtn.disabled = false;
    setStatus("Blueprint processed. Enable Autonomous Navigation Mode and click start/goal points.");
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  } finally {
    els.processBtn.disabled = false;
  }
}

/**
 * Handle point clicks in autonomous navigation mode.
 * @param {{x:number,y:number,z:number}} worldPoint
 */
async function onSceneClick(worldPoint) {
  if (!latestMeta) return;

  clickPoints.push(worldPoint);
  if (clickPoints.length === 1) {
    setStatus("Start point selected. Click destination point.");
    return;
  }

  if (clickPoints.length >= 2) {
    const [startWorld, goalWorld] = clickPoints.slice(-2);
    clickPoints = [];

    const start = worldToGrid({ x: startWorld.x, z: startWorld.z });
    const goal = worldToGrid({ x: goalWorld.x, z: goalWorld.z });

    setStatus("Computing path with A*...");

    try {
      const data = await navigator.fetchPath(BACKEND_BASE, start, goal);
      scene.setPath(data.world_path);
      navigator.animateAlongPath(data.world_path, 1.2);
      setStatus(`Path found with ${data.path.length} waypoints. Agent navigating.`);
    } catch (err) {
      setStatus(`Navigation failed: ${err.message}`);
    }
  }
}

/** Toggle autonomous navigation click mode. */
function toggleNavigationMode() {
  if (!latestMeta) {
    setStatus("Process a blueprint before enabling navigation mode.");
    return;
  }

  navMode = !navMode;
  scene.setClickMode(navMode, navMode ? onSceneClick : null);
  els.navModeBtn.textContent = navMode
    ? "Exit Autonomous Navigation Mode"
    : "Autonomous Navigation Mode";

  setStatus(navMode ? "Navigation mode enabled: click start then goal in the scene." : "Navigation mode disabled.");
}

els.processBtn.addEventListener("click", processBlueprint);
els.navModeBtn.addEventListener("click", toggleNavigationMode);
els.followCamera.addEventListener("change", (e) => {
  scene.setFollowAgent(Boolean(e.target.checked));
});
