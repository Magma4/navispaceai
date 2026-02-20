import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import SceneCanvas from "./scene/SceneCanvas";
import SceneCanvas3D from "./scene/SceneCanvas3D";
import FloorSelector from "./components/FloorSelector";
import NavigationPanel from "./components/NavigationPanel";
import ToastNotification from "./components/ToastNotification";
import { findPath, getFloors, getRooms } from "./api/backendAPI";
import { createAgentAnimator, gridPathToWorldPath } from "./scene/AgentAnimation";
import { parseGameURL } from "./utils/game-url";

/**
 * Dedicated standalone game runtime loaded in /game.html.
 *
 * Reverted layout:
 * - Top navigation bar
 * - Left sidebar controls
 * - Large render area
 */
export default function GameApp() {
  const config = useMemo(() => parseGameURL(window.location.href), []);

  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");

  const [grid, setGrid] = useState(null);
  const [floorsMeta, setFloorsMeta] = useState(null);
  const [rooms, setRooms] = useState([]);

  const [selectedFloor, setSelectedFloor] = useState("0");
  const [toast, setToast] = useState({ message: null, type: "info" });

  const [startCell, setStartCell] = useState(null);
  const [goalCell, setGoalCell] = useState(null);
  const [gridPath, setGridPath] = useState([]);
  const [worldPath, setWorldPath] = useState([]);

  const [agentPosition, setAgentPosition] = useState(null);
  const [agentYaw, setAgentYaw] = useState(0);
  const [agentMoving, setAgentMoving] = useState(false);
  const [agentSpeed, setAgentSpeed] = useState(1.2);

  const [autonomousEnabled, setAutonomousEnabled] = useState(true);
  const [cameraMode, setCameraMode] = useState("orbit");
  const [debugOverlay, setDebugOverlay] = useState(false);
  const [cameraPresetRequest, setCameraPresetRequest] = useState(null);
  const [rerouteCount, setRerouteCount] = useState(0);

  const animatorRef = useRef(null);
  const goalCellRef = useRef(null);
  const autonomousRef = useRef(true);
  const projectMetaRef = useRef(null);
  const replanInFlightRef = useRef(false);

  const sortedFloors = useMemo(() => {
    const raw = floorsMeta?.floors;
    if (!Array.isArray(raw)) return [];
    return [...raw].sort((a, b) => Number(a.floor_number) - Number(b.floor_number));
  }, [floorsMeta]);

  const isMultiFloor = sortedFloors.length > 0;

  const selectedFloorNumber = useMemo(() => {
    const parsed = Number(selectedFloor);
    return Number.isFinite(parsed) ? parsed : 0;
  }, [selectedFloor]);

  const activeFloorMeta = useMemo(() => {
    if (!isMultiFloor) return null;
    return (
      sortedFloors.find((floor) => Number(floor.floor_number) === Number(selectedFloorNumber)) ||
      sortedFloors[0] ||
      null
    );
  }, [isMultiFloor, selectedFloorNumber, sortedFloors]);

  const projectMeta = useMemo(() => {
    if (!Array.isArray(grid) || !grid.length) return null;
    return {
      grid,
      grid_shape: {
        rows: grid.length,
        cols: grid[0]?.length || 0,
      },
      cell_size_m: config.error ? 0.2 : config.cellSizeM,
      model_absolute_url: config.error ? "" : config.modelUrl,
    };
  }, [config, grid]);

  /**
   * Show toast message.
   * @param {string} message
   * @param {"error"|"success"|"info"} type
   */
  function showToast(message, type = "info") {
    setToast({ message, type });
  }

  function worldToGridCell(point, cellSizeM, gridShape) {
    if (!point || !cellSizeM || !gridShape) return null;
    return {
      row: Math.max(0, Math.min(gridShape.rows - 1, Math.round(point.z / cellSizeM))),
      col: Math.max(0, Math.min(gridShape.cols - 1, Math.round(point.x / cellSizeM))),
    };
  }

  useEffect(() => {
    goalCellRef.current = goalCell;
  }, [goalCell]);

  useEffect(() => {
    autonomousRef.current = autonomousEnabled;
  }, [autonomousEnabled]);

  useEffect(() => {
    projectMetaRef.current = projectMeta;
  }, [projectMeta]);

  /**
   * Stop and release active animator.
   */
  const stopAnimator = useCallback(() => {
    if (!animatorRef.current) return;
    animatorRef.current.stop();
    animatorRef.current = null;
    setAgentMoving(false);
  }, []);

  /**
   * Start or update autonomous movement animator.
   *
   * @param {Array<{x:number,z:number,y?:number}>} nextPath
   * @param {{preserveProgress?:boolean}} [options]
   */
  const runAnimator = useCallback(
    (nextPath, options = {}) => {
      if (!Array.isArray(nextPath) || nextPath.length < 2) {
        stopAnimator();
        return;
      }

      if (!animatorRef.current) {
        animatorRef.current = createAgentAnimator({
          initialPath: nextPath,
          initialSpeed: agentSpeed,
          occupancyGrid: projectMeta?.grid || null,
          cellSizeM: projectMeta?.cell_size_m || 0,
          onUpdate: (position, yaw) => {
            setAgentPosition({ x: position.x, z: position.z });
            setAgentYaw(yaw);
          },
          onCollision: async ({ position }) => {
            const goal = goalCellRef.current;
            const meta = projectMetaRef.current;
            if (!autonomousRef.current || !goal || !meta?.grid_shape || !meta?.cell_size_m) {
              showToast("Agent collision detected. Path stopped.", "error");
              return;
            }

            if (replanInFlightRef.current) return;
            replanInFlightRef.current = true;
            try {
              const start = worldToGridCell(position, meta.cell_size_m, meta.grid_shape);
              if (!start) throw new Error("Unable to replan: invalid collision position.");

              const response = await findPath(start, goal, true, config.backendBaseUrl);
              const rerouteGrid = response?.smoothed_path?.length ? response.smoothed_path : (response.path || []);
              const rerouteWorld = response?.smoothed_world_path?.length
                ? response.smoothed_world_path
                : (response.world_path || []);

              handlePathComputed({
                gridPath: rerouteGrid,
                worldPath: rerouteWorld,
                reason: "replan",
              });
              setRerouteCount((v) => v + 1);
              showToast("Obstacle hit. Auto re-routing complete.", "info");
            } catch (error) {
              showToast(error.message || "Auto re-routing failed.", "error");
            } finally {
              replanInFlightRef.current = false;
            }
          },
          onComplete: ({ reason }) => {
            setAgentMoving(false);
            if (reason === "completed") {
              showToast("Path complete.", "success");
            }
          },
        });
      } else {
        animatorRef.current.setPath(nextPath, options);
      }

      animatorRef.current.setSpeed(agentSpeed);
      animatorRef.current.start();
      setAgentMoving(true);
    },
    [agentSpeed, config.backendBaseUrl, projectMeta?.cell_size_m, projectMeta?.grid, stopAnimator]
  );

  /**
   * Initial game payload loading:
   * - load URL grid (single-floor)
   * - optionally probe multi-floor payload
   */
  useEffect(() => {
    if (config.error) {
      setLoading(false);
      setLoadError(config.error);
      return;
    }

    let active = true;
    setLoading(true);

    const load = async () => {
      let nextGrid = null;
      let nextFloors = null;
      let nextRooms = [];

      if (config.gridUrl) {
        try {
          const response = await fetch(config.gridUrl);
          if (!response.ok) throw new Error(`Could not load grid (${response.status}).`);
          const payload = await response.json();
          const parsedGrid = Array.isArray(payload) ? payload : payload?.grid;
          if (!Array.isArray(parsedGrid) || !parsedGrid.length || !Array.isArray(parsedGrid[0])) {
            throw new Error("Grid payload format is invalid.");
          }
          nextGrid = parsedGrid;
        } catch (error) {
          if (config.mode !== "multi") {
            throw new Error(error.message || "Failed to load game configuration.");
          }
        }
      }

      const shouldProbeMulti = config.mode === "multi" || !config.gridUrl;
      if (shouldProbeMulti) {
        try {
          const floorsPayload = await getFloors(config.backendBaseUrl);
          if (Array.isArray(floorsPayload?.floors) && floorsPayload.floors.length > 0) {
            nextFloors = floorsPayload;
            const roomsPayload = await getRooms(null, config.backendBaseUrl);
            nextRooms = Array.isArray(roomsPayload?.rooms) ? roomsPayload.rooms : [];
          }
        } catch (error) {
          if (config.mode === "multi") {
            throw new Error(error.message || "Failed to load multi-floor building metadata.");
          }
        }
      }

      if (!nextGrid && !nextFloors) {
        throw new Error("No game state available. Process a blueprint or building first.");
      }

      if (!active) return;
      setGrid(nextGrid);
      setFloorsMeta(nextFloors);
      setRooms(nextRooms);
      setLoadError("");
    };

    load()
      .catch((error) => {
        if (!active) return;
        setLoadError(error.message || "Failed to load game configuration.");
      })
      .finally(() => {
        if (!active) return;
        setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [config]);

  useEffect(() => {
    if (!isMultiFloor) {
      setSelectedFloor("0");
      return;
    }

    const exists = sortedFloors.some((floor) => String(floor.floor_number) === String(selectedFloor));
    if (!exists) {
      setSelectedFloor(String(sortedFloors[0].floor_number));
    }
  }, [isMultiFloor, selectedFloor, sortedFloors]);

  useEffect(() => {
    return () => stopAnimator();
  }, [stopAnimator]);

  useEffect(() => {
    if (!animatorRef.current) return;
    animatorRef.current.setSpeed(agentSpeed);
  }, [agentSpeed]);

  useEffect(() => {
    if (!autonomousEnabled) {
      stopAnimator();
      return;
    }
    if (worldPath.length >= 2) {
      runAnimator(worldPath, { preserveProgress: true });
    }
  }, [autonomousEnabled, worldPath, runAnimator, stopAnimator]);

  /**
   * Cycle start and goal via scene clicks:
   * - first click sets start
   * - second click sets goal
   * - third click resets selection and starts over
   *
   * @param {{cell:{row:number,col:number},world:{x:number,z:number}}} payload
   */
  function handleCellPick(payload) {
    if (isMultiFloor || !payload?.cell) return;

    const picked = payload.cell;
    if (!startCell || (startCell && goalCell)) {
      setStartCell(picked);
      setGoalCell(null);
      setGridPath([]);
      setWorldPath([]);
      setAgentPosition(null);
      setAgentYaw(0);
      stopAnimator();
      return;
    }
    setGoalCell(picked);
  }

  /**
   * Handle path results from control panel.
   *
   * @param {{gridPath:Array,worldPath:Array,reason:"compute"|"replan"}} payload
   */
  function handlePathComputed(payload) {
    const nextGridPath = Array.isArray(payload?.gridPath) ? payload.gridPath : [];
    const nextWorldPathRaw = Array.isArray(payload?.worldPath) ? payload.worldPath : [];
    const nextWorldPath =
      nextWorldPathRaw.length > 0
        ? nextWorldPathRaw.map((point) => ({ x: point.x, z: point.z, y: point.y ?? 0.2 }))
        : gridPathToWorldPath(nextGridPath, projectMeta?.cell_size_m || 0.2);

    setGridPath(nextGridPath);
    setWorldPath(nextWorldPath);
    if (payload?.reason === "compute") {
      setRerouteCount(0);
    }

    if (nextWorldPath.length > 0) {
      const first = nextWorldPath[0];
      setAgentPosition({ x: first.x, z: first.z });
      if (nextWorldPath.length > 1) {
        const second = nextWorldPath[1];
        setAgentYaw(Math.atan2(second.z - first.z, second.x - first.x));
      }
    }

    if (autonomousEnabled && nextWorldPath.length >= 2) {
      runAnimator(nextWorldPath, { preserveProgress: payload?.reason === "replan" });
    } else {
      stopAnimator();
    }
  }

  /**
   * Clear current route and movement state.
   */
  function handleClearRoute() {
    setGridPath([]);
    setWorldPath([]);
    setAgentPosition(null);
    setAgentYaw(0);
    setRerouteCount(0);
    stopAnimator();
  }

  /**
   * Proxy path requests from NavigationPanel.
   *
   * @param {{start:{row:number,col:number},goal:{row:number,col:number}}} params
   */
  async function requestPath(params) {
    const response = await findPath(params.start, params.goal, true, config.backendBaseUrl);
    return {
      ...response,
      path: response?.smoothed_path?.length ? response.smoothed_path : (response?.path || []),
      world_path: response?.smoothed_world_path?.length
        ? response.smoothed_world_path
        : (response?.world_path || []),
    };
  }

  /**
   * Trigger camera preset transitions.
   * @param {"fit"|"aerial"} mode
   */
  function requestCameraPreset(mode) {
    setCameraPresetRequest({ mode, nonce: Date.now() });
  }

  if (loading) {
    return (
      <div className="app game-app">
        <div className="game-bg" aria-hidden="true">
          <span className="orb orb-a" />
          <span className="orb orb-b" />
          <span className="grid-glow" />
        </div>
        <header className="topbar game-topbar">
          <div className="brand-wrap game-brand">
            <div className="brand-icon game-brand-icon" aria-hidden="true">
              N
            </div>
            <div>
              <h1>NavispaceAI Game</h1>
              <p>Loading standalone runtime...</p>
            </div>
          </div>
        </header>
      </div>
    );
  }

  const canRenderSingle = !!projectMeta;
  const canRenderMulti = isMultiFloor;

  if (loadError || (!canRenderSingle && !canRenderMulti)) {
    return (
      <div className="app game-app">
        <header className="topbar game-topbar">
          <div className="brand-wrap game-brand">
            <div className="brand-icon game-brand-icon" aria-hidden="true">
              N
            </div>
            <div>
              <h1>NavispaceAI Game</h1>
              <p>Game configuration could not be loaded</p>
            </div>
          </div>
        </header>

        <main className="layout sidebar-hidden game-layout">
          <section className="panel game-panel" style={{ maxWidth: 760 }}>
            <h2>Unable to Start Game</h2>
            <p className="muted">{loadError || "Unknown game loading error."}</p>
            <div className="action-row">
              <a className="btn btn-primary" href="/">
                Back to Upload Dashboard
              </a>
            </div>
          </section>
        </main>
      </div>
    );
  }

  const gridRows = isMultiFloor ? Number(activeFloorMeta?.rows || 0) : projectMeta.grid_shape.rows;
  const gridCols = isMultiFloor ? Number(activeFloorMeta?.cols || 0) : projectMeta.grid_shape.cols;

  const floorOptions = isMultiFloor
    ? sortedFloors.map((floor) => ({
        label: `Floor ${floor.floor_number}`,
        value: String(floor.floor_number),
      }))
    : [{ label: "Floor 0", value: "0" }];

  return (
    <div className="app game-app">
      <div className="game-bg" aria-hidden="true">
        <span className="orb orb-a" />
        <span className="orb orb-b" />
        <span className="orb orb-c" />
        <span className="grid-glow" />
      </div>

      <header className="topbar game-topbar">
        <div className="brand-wrap game-brand">
          <div className="brand-icon game-brand-icon" aria-hidden="true">
            N
          </div>
          <div>
            <h1>NavispaceAI Game</h1>
            <p>Standalone 3D navigation runtime</p>
          </div>
        </div>

        <div className="topbar-metrics game-metrics">
          <span className="status-pill is-ready">Game Active</span>
          <span className="status-pill">Grid: {gridRows} x {gridCols}</span>
          <span className="backend-pill">Backend: {config.backendBaseUrl}</span>
        </div>
      </header>

      <main className="layout sidebar-visible game-layout">
        <aside className="sidebar game-sidebar" aria-label="Game controls sidebar">
          <section className="panel">
            <h2>Game Views</h2>
            <div className="action-row">
              <button className="btn btn-secondary" type="button" onClick={() => requestCameraPreset("fit")}>
                Reset View
              </button>
              <button className="btn btn-secondary" type="button" onClick={() => requestCameraPreset("aerial")}>
                Aerial View
              </button>
            </div>
            <div className="action-row">
              <a className="btn btn-ghost" href="/">
                Open Upload Dashboard
              </a>
            </div>
          </section>

          <FloorSelector floors={floorOptions} value={selectedFloor} onChange={setSelectedFloor} />

          {!isMultiFloor ? (
            <NavigationPanel
              backendBaseUrl={config.backendBaseUrl}
              gridShape={projectMeta.grid_shape}
              startCell={startCell}
              goalCell={goalCell}
              onStartChange={setStartCell}
              onGoalChange={setGoalCell}
              onPathComputed={handlePathComputed}
              onClear={handleClearRoute}
              onNotify={showToast}
              isAgentMoving={agentMoving}
              autonomousEnabled={autonomousEnabled}
              onAutonomousToggle={setAutonomousEnabled}
              cameraMode={cameraMode}
              onCameraModeChange={setCameraMode}
              agentSpeed={agentSpeed}
              onAgentSpeedChange={setAgentSpeed}
              debugOverlay={debugOverlay}
              onDebugOverlayChange={setDebugOverlay}
              activeFloor={`Floor ${selectedFloorNumber}`}
              pathRequest={requestPath}
            />
          ) : (
            <section className="panel">
              <h2>Navigation</h2>
              <p className="muted">
                Multi-floor runtime is active. Room-to-room and vertical connector navigation is available from the
                dashboard flow.
              </p>
            </section>
          )}
        </aside>

        <section className="scene-area game-scene-area">
          <div className="scene-header game-scene-header" aria-live="polite">
            <div className="scene-header-main">
              <strong>
                {isMultiFloor
                  ? "Multi-floor 3D scene loaded."
                  : "Click in the scene to pick start and goal cells."}
              </strong>
              <span className="muted">
                {isMultiFloor
                  ? "This tab is the standalone game runtime users can access directly."
                  : "Tip: third click resets selection and starts a new route."}
              </span>
            </div>
            <div className="scene-badges">
              <span className="status-pill">Mode: {isMultiFloor ? "Multi-floor" : "Single-floor"}</span>
              <span className="status-pill">Waypoints: {gridPath.length}</span>
              <span className="status-pill">Re-routes: {rerouteCount}</span>
              <span className="status-pill">Auto: {autonomousEnabled ? "ON" : "OFF"}</span>
              <span className="status-pill">Floor: {`Floor ${selectedFloorNumber}`}</span>
            </div>
          </div>

          {isMultiFloor ? (
            <SceneCanvas3D
              floors={sortedFloors}
              rooms={rooms}
              activeFloor={selectedFloorNumber}
              path={[]}
              agentPosition={null}
              agentYaw={0}
              cameraMode={cameraMode}
              onWorldPick={() => {}}
            />
          ) : (
            <SceneCanvas
              modelUrl={config.modelUrl}
              occupancyGrid={projectMeta.grid}
              cellSizeM={projectMeta.cell_size_m}
              startCell={startCell}
              goalCell={goalCell}
              pathWorld={worldPath}
              agentPosition={agentPosition}
              agentYaw={agentYaw}
              followCamera={cameraMode === "follow"}
              cameraPresetRequest={cameraPresetRequest}
              onCellPick={handleCellPick}
              immersiveMode={false}
              thirdPersonEnabled={false}
              initialPlayerPosition={null}
            />
          )}
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
