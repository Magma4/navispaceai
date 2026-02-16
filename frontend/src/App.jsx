import { useCallback, useMemo, useRef, useState } from "react";

import UploadForm from "./components/UploadForm";
import NavigationPanel from "./components/NavigationPanel";
import AutonomousModeToggle from "./components/AutonomousModeToggle";
import FloorSelector from "./components/FloorSelector";
import ToastNotification from "./components/ToastNotification";
import SceneCanvas from "./scene/SceneCanvas";
import { animateAgentAlongPath } from "./scene/AgentAnimation";
import {
  BACKEND_BASE_URL,
  findPath,
  resolveBackendURL,
} from "./api/backendAPI";

/**
 * App is the main NavispaceAI frontend container.
 * It coordinates upload, backend processing, point picking, pathfinding, and animation.
 */
export default function App() {
  const [processing, setProcessing] = useState(false);
  const [projectMeta, setProjectMeta] = useState(null);
  const [selectedFloor, setSelectedFloor] = useState("Floor 1");
  const [toast, setToast] = useState({ message: null, type: "info" });

  const [startCell, setStartCell] = useState(null);
  const [goalCell, setGoalCell] = useState(null);

  const [pathWorld, setPathWorld] = useState([]);
  const [autonomousMode, setAutonomousMode] = useState(false);
  const [followCamera, setFollowCamera] = useState(false);
  const [agentSpeed, setAgentSpeed] = useState(1.2);

  const [agentPosition, setAgentPosition] = useState(null);
  const [agentYaw, setAgentYaw] = useState(0);

  const cancelAnimationRef = useRef(() => { });

  const floors = useMemo(() => ["Floor 1", "Floor 2", "Floor 3"], []);

  /**
   * Show toast notification message.
   * @param {string} message
   * @param {"error"|"success"|"info"} type
   */
  function showToast(message, type = "info") {
    setToast({ message, type });
  }

  /**
   * Clear all selected points and path data.
   */
  const clearNavigation = useCallback(() => {
    setStartCell(null);
    setGoalCell(null);
    setPathWorld([]);
    setAgentPosition(null);
    setAgentYaw(0);
    cancelAnimationRef.current?.();
  }, []);

  /**
   * Convert world-space click point to grid coordinates using process metadata.
   * @param {{x:number,z:number}} worldPoint
   * @returns {{row:number,col:number}|null}
   */
  function worldToGrid(worldPoint) {
    if (!projectMeta) return null;

    const rows = projectMeta.grid_shape.rows;
    const cols = projectMeta.grid_shape.cols;
    const cellSize = projectMeta.cell_size_m;

    const row = Math.max(0, Math.min(rows - 1, Math.round(worldPoint.z / cellSize)));
    const col = Math.max(0, Math.min(cols - 1, Math.round(worldPoint.x / cellSize)));

    return { row, col };
  }

  /** Handle successful backend processing payload from UploadForm. */
  function handleProcessed(result) {
    clearNavigation();
    setProjectMeta(result);
  }

  /**
   * Start autonomous agent animation from currently loaded path.
   * This function safely cancels previous animation before starting a new one.
   */
  function runAutonomousAnimation() {
    cancelAnimationRef.current?.();

    cancelAnimationRef.current = animateAgentAlongPath({
      worldPath: pathWorld,
      speedMps: agentSpeed,
      onUpdate: (position, yaw) => {
        setAgentPosition(position);
        setAgentYaw(yaw);
      },
      onComplete: () => {
        showToast("Destination reached.", "success");
      },
    });
  }

  /**
   * Handle clicks on the scene for start/goal selection and path requests.
   * @param {{x:number,z:number}} worldPoint
   */
  async function handleScenePointPick(worldPoint) {
    if (!projectMeta) return;

    const cell = worldToGrid(worldPoint);
    if (!cell) return;

    if (!startCell) {
      setStartCell(cell);
      showToast("Start point selected. Choose goal point.", "info");
      return;
    }

    if (!goalCell) {
      setGoalCell(cell);

      try {
        const response = await findPath(startCell, cell, true, BACKEND_BASE_URL);
        setPathWorld(response.world_path || []);

        if (response.world_path?.length) {
          const first = response.world_path[0];
          setAgentPosition({ x: first.x, z: first.z });
          setAgentYaw(0);
        }

        showToast(`Path computed (${response.path.length} waypoints).`, "success");

        if (autonomousMode) {
          runAutonomousAnimation();
        }
      } catch (error) {
        showToast(error.message || "Pathfinding failed.", "error");
      }
      return;
    }

    // If both points exist, third click resets with a new start.
    clearNavigation();
    setStartCell(cell);
    showToast("Selection reset. New start point selected.", "info");
  }

  /**
   * Handle path results from NavigationPanel (manual compute/replan).
   * @param {{gridPath:Array,worldPath:Array,reason?:string}} payload
   */
  function handlePathComputed(payload) {
    const worldPath = payload?.worldPath || [];
    setPathWorld(worldPath);
    if (worldPath.length) {
      setAgentPosition({ x: worldPath[0].x, z: worldPath[0].z });
      setAgentYaw(0);
      if (autonomousMode) runAutonomousAnimation();
    }
  }

  /**
   * Toggle autonomous mode and optionally start animation if path is available.
   * @param {boolean} enabled
   */
  function handleAutonomousToggle(enabled) {
    setAutonomousMode(enabled);

    if (!enabled) {
      cancelAnimationRef.current?.();
      showToast("Autonomous mode disabled.", "info");
      return;
    }

    if (!pathWorld.length) {
      showToast("Compute a path before enabling autonomous mode.", "error");
      setAutonomousMode(false);
      return;
    }

    runAutonomousAnimation();
    showToast("Autonomous mode enabled.", "success");
  }

  const resolvedModelURL = resolveBackendURL(
    projectMeta?.model_absolute_url || projectMeta?.model_url,
    BACKEND_BASE_URL
  );
  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>NavispaceAI</h1>
          <p>Blueprint to 3D navigation workspace</p>
        </div>
        <span className="backend-pill">Backend: {BACKEND_BASE_URL}</span>
      </header>

      <main className="layout">
        <aside className="sidebar" aria-label="Control sidebar">
          <UploadForm
            backendBaseUrl={BACKEND_BASE_URL}
            onProcessed={handleProcessed}
            onNotify={showToast}
            onProcessingChange={setProcessing}
          />

          <FloorSelector floors={floors} value={selectedFloor} onChange={setSelectedFloor} />

          <NavigationPanel
            backendBaseUrl={BACKEND_BASE_URL}
            gridShape={projectMeta?.grid_shape || null}
            startCell={startCell}
            goalCell={goalCell}
            onStartChange={setStartCell}
            onGoalChange={setGoalCell}
            onPathComputed={handlePathComputed}
            onClear={clearNavigation}
            onNotify={showToast}
            isAgentMoving={Boolean(agentPosition)}
            autonomousEnabled={autonomousMode}
            onAutonomousToggle={handleAutonomousToggle}
            cameraMode={followCamera ? "follow" : "orbit"}
            onCameraModeChange={(mode) => setFollowCamera(mode === "follow")}
            agentSpeed={agentSpeed}
            onAgentSpeedChange={setAgentSpeed}
            debugOverlay={false}
            onDebugOverlayChange={() => {}}
            activeFloor={selectedFloor}
          />

          <AutonomousModeToggle
            enabled={autonomousMode}
            disabled={!projectMeta || !pathWorld.length}
            onToggle={handleAutonomousToggle}
            followCamera={followCamera}
            onFollowCameraChange={setFollowCamera}
          />
        </aside>

        <section className="scene-area">
          <div className="scene-header" aria-live="polite">
            {!projectMeta
              ? "Upload and process a blueprint to begin."
              : "Click in the scene to pick start and goal cells."}
          </div>

          <SceneCanvas
            modelUrl={resolvedModelURL}
            occupancyGrid={projectMeta?.grid || null}
            cellSizeM={projectMeta?.cell_size_m || 0.2}
            startCell={startCell}
            goalCell={goalCell}
            pathWorld={pathWorld}
            agentPosition={agentPosition}
            agentYaw={agentYaw}
            followCamera={followCamera}
            onCellPick={(payload) => handleScenePointPick(payload.world)}
          />
        </section>
      </main>

      <ToastNotification
        message={toast.message}
        type={toast.type}
        onClose={() => setToast({ message: null, type: "info" })}
      />
    </div>
  );
}
