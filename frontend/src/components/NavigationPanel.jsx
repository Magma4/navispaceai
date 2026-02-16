import { useEffect, useMemo, useState } from "react";

import { BACKEND_BASE_URL, findPath } from "../api/backendAPI";

/**
 * NavigationPanel controls path requests and game-mode runtime settings.
 *
 * Added UX controls:
 * - Autonomous mode toggle
 * - Camera mode toggle (orbit/follow)
 * - Agent speed slider
 * - Debug overlay toggle
 * - Re-plan on goal change while moving
 *
 * @param {object} props
 * @param {string} [props.backendBaseUrl]
 * @param {{rows:number,cols:number}|null} props.gridShape
 * @param {{row:number,col:number}|null} props.startCell
 * @param {{row:number,col:number}|null} props.goalCell
 * @param {(start:{row:number,col:number}|null)=>void} props.onStartChange
 * @param {(goal:{row:number,col:number}|null)=>void} props.onGoalChange
 * @param {(data:{gridPath:Array,worldPath:Array,reason:"compute"|"replan"})=>void} props.onPathComputed
 * @param {() => void} [props.onClear]
 * @param {(message:string,type?:"error"|"success"|"info")=>void} [props.onNotify]
 * @param {boolean} [props.isAgentMoving=false]
 * @param {boolean} [props.autonomousEnabled=false]
 * @param {(next:boolean)=>void} [props.onAutonomousToggle]
 * @param {"orbit"|"follow"} [props.cameraMode="orbit"]
 * @param {(mode:"orbit"|"follow")=>void} [props.onCameraModeChange]
 * @param {number} [props.agentSpeed=1.2]
 * @param {(speed:number)=>void} [props.onAgentSpeedChange]
 * @param {boolean} [props.debugOverlay=false]
 * @param {(next:boolean)=>void} [props.onDebugOverlayChange]
 * @param {string|number} [props.activeFloor="Floor 1"]
 */
export default function NavigationPanel({
  backendBaseUrl = BACKEND_BASE_URL,
  gridShape,
  startCell,
  goalCell,
  onStartChange,
  onGoalChange,
  onPathComputed,
  onClear,
  onNotify,
  isAgentMoving = false,
  autonomousEnabled = false,
  onAutonomousToggle,
  cameraMode = "orbit",
  onCameraModeChange,
  agentSpeed = 1.2,
  onAgentSpeedChange,
  debugOverlay = false,
  onDebugOverlayChange,
  activeFloor = "Floor 1",
}) {
  const [startRow, setStartRow] = useState("");
  const [startCol, setStartCol] = useState("");
  const [goalRow, setGoalRow] = useState("");
  const [goalCol, setGoalCol] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!startCell) return;
    setStartRow(String(startCell.row));
    setStartCol(String(startCell.col));
  }, [startCell]);

  useEffect(() => {
    if (!goalCell) return;
    setGoalRow(String(goalCell.row));
    setGoalCol(String(goalCell.col));
  }, [goalCell]);

  /**
   * Parse input values into grid cell.
   * @param {string} row
   * @param {string} col
   * @returns {{row:number,col:number}|null}
   */
  function parseCell(row, col) {
    if (row === "" || col === "") return null;
    const parsedRow = Number(row);
    const parsedCol = Number(col);
    if (!Number.isInteger(parsedRow) || !Number.isInteger(parsedCol)) return null;
    return { row: parsedRow, col: parsedCol };
  }

  const parsedStart = useMemo(() => parseCell(startRow, startCol), [startRow, startCol]);
  const parsedGoal = useMemo(() => parseCell(goalRow, goalCol), [goalRow, goalCol]);

  /**
   * Check cell inside occupancy bounds.
   * @param {{row:number,col:number}|null} cell
   * @returns {boolean}
   */
  function inBounds(cell) {
    if (!gridShape || !cell) return false;
    return cell.row >= 0 && cell.col >= 0 && cell.row < gridShape.rows && cell.col < gridShape.cols;
  }

  /**
   * Request A* path from backend.
   * @param {"compute"|"replan"} reason
   */
  async function requestPath(reason = "compute") {
    if (!parsedStart || !parsedGoal || !gridShape) return;

    if (!inBounds(parsedStart) || !inBounds(parsedGoal)) {
      onNotify?.("Start or goal is outside the occupancy grid.", "error");
      return;
    }

    setLoading(true);
    try {
      onStartChange?.(parsedStart);
      onGoalChange?.(parsedGoal);

      const response = await findPath(parsedStart, parsedGoal, true, backendBaseUrl);
      onPathComputed?.({
        gridPath: response.path || [],
        worldPath: response.world_path || [],
        reason,
      });

      if (reason === "replan") {
        onNotify?.(`Path re-planned (${response.path?.length || 0} waypoints).`, "info");
      } else {
        onNotify?.(`Path found (${response.path?.length || 0} waypoints).`, "success");
      }
    } catch (error) {
      onNotify?.(error.message || "Failed to compute path.", "error");
    } finally {
      setLoading(false);
    }
  }

  /**
   * Trigger immediate re-plan when goal is edited while agent is moving.
   */
  useEffect(() => {
    if (!isAgentMoving || !autonomousEnabled) return;
    if (!parsedStart || !parsedGoal) return;
    if (!gridShape || loading) return;

    const timer = setTimeout(() => {
      requestPath("replan");
    }, 250);

    return () => clearTimeout(timer);
  }, [
    goalRow,
    goalCol,
    isAgentMoving,
    autonomousEnabled,
    parsedStart,
    parsedGoal,
    gridShape,
    loading,
  ]);

  /**
   * Reset pathfinding and movement related local values.
   */
  function handleClear() {
    setStartRow("");
    setStartCol("");
    setGoalRow("");
    setGoalCol("");
    onStartChange?.(null);
    onGoalChange?.(null);
    onClear?.();
  }

  const computeDisabled = !gridShape || !parsedStart || !parsedGoal || loading;

  return (
    <section className="panel nav-panel" aria-label="Navigation controls">
      <h2>Navigation Game Controls</h2>
      <p className="muted">Floor: {activeFloor} (multi-floor vertical movement placeholder ready)</p>

      <div className="grid-inputs">
        <div className="cell-card">
          <h3>Start</h3>
          <label>
            Row
            <input
              type="number"
              min={0}
              value={startRow}
              disabled={loading}
              onChange={(event) => setStartRow(event.target.value)}
            />
          </label>
          <label>
            Col
            <input
              type="number"
              min={0}
              value={startCol}
              disabled={loading}
              onChange={(event) => setStartCol(event.target.value)}
            />
          </label>
        </div>

        <div className="cell-card">
          <h3>Goal</h3>
          <label>
            Row
            <input
              type="number"
              min={0}
              value={goalRow}
              disabled={loading}
              onChange={(event) => setGoalRow(event.target.value)}
            />
          </label>
          <label>
            Col
            <input
              type="number"
              min={0}
              value={goalCol}
              disabled={loading}
              onChange={(event) => setGoalCol(event.target.value)}
            />
          </label>
        </div>
      </div>

      <div className="muted" aria-live="polite" style={{ marginBottom: 8 }}>
        {gridShape ? `Grid: ${gridShape.rows} x ${gridShape.cols}` : "Process a blueprint to load occupancy grid."}
      </div>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 10 }}>
        <button className="btn btn-primary" type="button" onClick={() => requestPath("compute")} disabled={computeDisabled}>
          {loading ? "Computing..." : "Compute A* Path"}
        </button>
        <button className="btn btn-secondary" type="button" onClick={handleClear} disabled={loading}>
          Clear
        </button>
      </div>

      <hr style={{ border: "none", borderTop: "1px solid var(--border)", margin: "10px 0" }} />

      <div style={{ display: "grid", gap: 10 }}>
        <label className="checkbox" style={{ marginTop: 0 }}>
          <input
            type="checkbox"
            checked={autonomousEnabled}
            onChange={(event) => onAutonomousToggle?.(event.target.checked)}
          />
          <span>Autonomous Mode</span>
        </label>

        <label>
          <span className="muted">Camera Mode</span>
          <select
            className="select"
            value={cameraMode}
            onChange={(event) => onCameraModeChange?.(event.target.value)}
          >
            <option value="orbit">Orbit Mode</option>
            <option value="follow">Follow Mode</option>
          </select>
        </label>

        <label>
          <span className="muted">Agent Speed: {Number(agentSpeed).toFixed(2)} m/s</span>
          <input
            type="range"
            min="0.3"
            max="4.0"
            step="0.1"
            value={agentSpeed}
            onChange={(event) => onAgentSpeedChange?.(Number(event.target.value))}
          />
        </label>

        <label className="checkbox" style={{ marginTop: 0 }}>
          <input
            type="checkbox"
            checked={debugOverlay}
            onChange={(event) => onDebugOverlayChange?.(event.target.checked)}
          />
          <span>Debug Overlay (collisions)</span>
        </label>
      </div>
    </section>
  );
}
