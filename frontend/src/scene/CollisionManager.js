/**
 * CollisionManager centralizes occupancy-grid based collision checks.
 *
 * The backend grid contract is:
 * - 0 => free
 * - 1 => occupied
 */

/**
 * Convert world-space position to clamped grid coordinate.
 *
 * @param {{x:number,z:number}} position
 * @param {number} cellSizeM
 * @param {number[][]} occupancyGrid
 * @returns {{row:number,col:number}|null}
 */
export function worldToGridCell(position, cellSizeM, occupancyGrid) {
  if (!position || !Array.isArray(occupancyGrid) || occupancyGrid.length === 0 || !cellSizeM) {
    return null;
  }

  const rows = occupancyGrid.length;
  const cols = occupancyGrid[0]?.length || 0;
  if (!rows || !cols) return null;

  const row = Math.max(0, Math.min(rows - 1, Math.round(position.z / cellSizeM)));
  const col = Math.max(0, Math.min(cols - 1, Math.round(position.x / cellSizeM)));

  return { row, col };
}

/**
 * Convert grid cell to world-space center point.
 *
 * @param {{row:number,col:number}} cell
 * @param {number} cellSizeM
 * @returns {{x:number,z:number}}
 */
export function gridCellToWorld(cell, cellSizeM) {
  return {
    x: cell.col * cellSizeM,
    z: cell.row * cellSizeM,
  };
}

/**
 * Check if a given grid cell is occupied.
 *
 * @param {{row:number,col:number}|null} cell
 * @param {number[][]} occupancyGrid
 * @returns {boolean}
 */
export function isCellOccupied(cell, occupancyGrid) {
  if (!cell || !Array.isArray(occupancyGrid) || !occupancyGrid.length) return false;

  const rows = occupancyGrid.length;
  const cols = occupancyGrid[0]?.length || 0;
  if (rows === 0 || cols === 0) return false;

  if (cell.row < 0 || cell.col < 0 || cell.row >= rows || cell.col >= cols) return true;
  return occupancyGrid[cell.row]?.[cell.col] === 1;
}

/**
 * Check if a world-space position is colliding with occupied cell.
 *
 * @param {{x:number,z:number}} position
 * @param {number[][]} occupancyGrid
 * @param {number} cellSizeM
 * @returns {boolean}
 */
export function isWorldPositionBlocked(position, occupancyGrid, cellSizeM) {
  const cell = worldToGridCell(position, cellSizeM, occupancyGrid);
  return isCellOccupied(cell, occupancyGrid);
}

/**
 * Sample along a segment and detect first collision point.
 *
 * @param {{x:number,z:number}} from
 * @param {{x:number,z:number}} to
 * @param {number[][]} occupancyGrid
 * @param {number} cellSizeM
 * @param {number} [samples=10]
 * @returns {{hit:boolean,position?:{x:number,z:number},cell?:{row:number,col:number}}}
 */
export function segmentCollisionCheck(from, to, occupancyGrid, cellSizeM, samples = 10) {
  if (!from || !to) return { hit: false };

  const n = Math.max(2, samples);
  for (let i = 1; i <= n; i += 1) {
    const t = i / n;
    const x = from.x + (to.x - from.x) * t;
    const z = from.z + (to.z - from.z) * t;
    const probe = { x, z };

    if (isWorldPositionBlocked(probe, occupancyGrid, cellSizeM)) {
      return {
        hit: true,
        position: probe,
        cell: worldToGridCell(probe, cellSizeM, occupancyGrid),
      };
    }
  }

  return { hit: false };
}

/**
 * Build debug marker points for occupied cells (sparse for performance).
 *
 * @param {number[][]} occupancyGrid
 * @param {number} cellSizeM
 * @param {number} [step=4] - Sampling step for large grids.
 * @returns {Array<{x:number,z:number}>}
 */
export function buildCollisionDebugPoints(occupancyGrid, cellSizeM, step = 4) {
  if (!Array.isArray(occupancyGrid) || !occupancyGrid.length || !cellSizeM) return [];

  const points = [];
  const rows = occupancyGrid.length;
  const cols = occupancyGrid[0]?.length || 0;

  for (let r = 0; r < rows; r += step) {
    for (let c = 0; c < cols; c += step) {
      if (occupancyGrid[r]?.[c] === 1) {
        points.push({ x: c * cellSizeM, z: r * cellSizeM });
      }
    }
  }

  return points;
}
