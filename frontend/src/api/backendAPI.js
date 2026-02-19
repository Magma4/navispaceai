/**
 * Centralized FastAPI client for NavispaceAI frontend.
 *
 * Endpoints:
 * - POST /process-blueprint
 * - POST /find-path
 * - POST /process-building
 * - GET /floors
 * - GET /rooms
 * - POST /find-path-3d
 */

export const BACKEND_BASE_URL =
  import.meta.env.VITE_API_URL ||
  import.meta.env.REACT_APP_API_URL ||
  "http://localhost:8000";

/**
 * Join backend base URL and route path.
 * @param {string} baseUrl
 * @param {string} path
 * @returns {string}
 */
function buildURL(baseUrl, path) {
  const normalizedBase = String(baseUrl || BACKEND_BASE_URL).replace(/\/$/, "");
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${normalizedBase}${normalizedPath}`;
}

/**
 * Read backend error details and throw normalized Error.
 * @param {Response} response
 * @returns {Promise<never>}
 */
async function throwAPIError(response) {
  const fallback = `Request failed (${response.status})`;

  let message = fallback;
  try {
    const data = await response.json();
    message = data?.detail || data?.message || fallback;
  } catch {
    // Keep fallback message when body is not JSON.
  }

  throw new Error(message);
}

/**
 * Resolve backend-relative file URL to an absolute URL.
 * @param {string} maybeRelativeURL
 * @param {string} [baseUrl=BACKEND_BASE_URL]
 * @returns {string}
 */
export function resolveBackendURL(maybeRelativeURL, baseUrl = BACKEND_BASE_URL) {
  if (!maybeRelativeURL) return "";
  if (/^https?:\/\//i.test(maybeRelativeURL)) return maybeRelativeURL;
  return buildURL(baseUrl, maybeRelativeURL);
}

/**
 * Upload blueprint and run backend processing pipeline.
 *
 * @param {File} file
 * @param {string} [baseUrl=BACKEND_BASE_URL]
 * @returns {Promise<{
 *   message:string,
 *   walls:Array,
 *   doors:Array,
 *   grid:number[][],
 *   grid_shape:{rows:number,cols:number},
 *   cell_size_m:number,
 *   model_url:string,
 *   grid_url:string,
 *   model_absolute_url:string
 * }>} Processed payload.
 */
export async function processBlueprint(file, baseUrl = BACKEND_BASE_URL) {
  if (!file) {
    throw new Error("Blueprint file is required.");
  }

  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(buildURL(baseUrl, "/process-blueprint"), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    await throwAPIError(response);
  }

  const payload = await response.json();
  return {
    ...payload,
    model_absolute_url: resolveBackendURL(payload.model_url, baseUrl),
  };
}

/**
 * Upload multiple floor blueprints and initialize multi-floor building state.
 *
 * @param {File[]} files
 * @param {object} [options]
 * @param {number[]} [options.floorNumbers]
 * @param {string} [options.buildingId]
 * @param {number} [options.floorHeightM]
 * @param {number} [options.cellSizeM]
 * @param {Array<object>} [options.connectors]
 * @param {string} [baseUrl=BACKEND_BASE_URL]
 * @returns {Promise<object>}
 */
export async function processBuilding(files, options = {}, baseUrl = BACKEND_BASE_URL) {
  if (!Array.isArray(files) || files.length === 0) {
    throw new Error("At least one blueprint file is required.");
  }

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  if (Array.isArray(options.floorNumbers)) {
    formData.append("floor_numbers", JSON.stringify(options.floorNumbers));
  }

  if (options.buildingId) formData.append("building_id", String(options.buildingId));
  if (options.floorHeightM) formData.append("floor_height_m", String(options.floorHeightM));
  if (options.cellSizeM) formData.append("cell_size_m", String(options.cellSizeM));
  if (Array.isArray(options.connectors)) {
    formData.append("connector_json", JSON.stringify(options.connectors));
  }

  const response = await fetch(buildURL(baseUrl, "/process-building"), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    await throwAPIError(response);
  }

  const payload = await response.json();
  const floors = Array.isArray(payload.floors)
    ? payload.floors.map((floor) => ({
        ...floor,
        model_absolute_url: resolveBackendURL(floor.model_url, baseUrl),
      }))
    : [];

  return {
    ...payload,
    floors,
  };
}

/**
 * Request A* path for start/goal grid cells.
 *
 * @param {{row:number,col:number}} start
 * @param {{row:number,col:number}} goal
 * @param {boolean} [diagonal=true]
 * @param {string} [baseUrl=BACKEND_BASE_URL]
 * @returns {Promise<{path:Array<{row:number,col:number}>,world_path:Array<{x:number,z:number}>}>}
 */
export async function findPath(start, goal, diagonal = true, baseUrl = BACKEND_BASE_URL) {
  if (!start || !goal) {
    throw new Error("Start and goal are required.");
  }

  const response = await fetch(buildURL(baseUrl, "/find-path"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ start, goal, diagonal }),
  });

  if (!response.ok) {
    await throwAPIError(response);
  }

  return response.json();
}

/**
 * Fetch building and floor metadata for latest multi-floor build.
 *
 * @param {string} [baseUrl=BACKEND_BASE_URL]
 * @returns {Promise<{
 *   building_id:string,
 *   origin_m:[number,number,number],
 *   floor_height_m:number,
 *   connectors:Array<object>,
 *   floors:Array<object>
 * }>}
 */
export async function getFloors(baseUrl = BACKEND_BASE_URL) {
  const response = await fetch(buildURL(baseUrl, "/floors"));
  if (!response.ok) {
    await throwAPIError(response);
  }

  const payload = await response.json();
  return {
    ...payload,
    floors: Array.isArray(payload.floors)
      ? payload.floors.map((floor) => ({
          ...floor,
          model_absolute_url: resolveBackendURL(floor.model_url, baseUrl),
        }))
      : [],
  };
}

/**
 * Fetch room metadata. Optional floor filtering is supported.
 *
 * @param {number|null} [floorNumber=null]
 * @param {string} [baseUrl=BACKEND_BASE_URL]
 * @returns {Promise<{building_id:string, rooms:Array<object>}>}
 */
export async function getRooms(floorNumber = null, baseUrl = BACKEND_BASE_URL) {
  const path =
    floorNumber === null || floorNumber === undefined
      ? "/rooms"
      : `/rooms?floor_number=${encodeURIComponent(floorNumber)}`;
  const response = await fetch(buildURL(baseUrl, path));
  if (!response.ok) {
    await throwAPIError(response);
  }
  return response.json();
}

/**
 * Request 3D A* path in world meters or room IDs.
 *
 * @param {{
 *   start?:{x:number,y:number,z:number},
 *   goal?:{x:number,y:number,z:number},
 *   start_room_id?:string,
 *   goal_room_id?:string
 * }} payload
 * @param {string} [baseUrl=BACKEND_BASE_URL]
 * @returns {Promise<{grid_path:Array<{floor:number,row:number,col:number}>, world_path:Array<{x:number,y:number,z:number}>}>}
 */
export async function findPath3D(payload, baseUrl = BACKEND_BASE_URL) {
  if (!payload || typeof payload !== "object") {
    throw new Error("findPath3D payload is required.");
  }

  const response = await fetch(buildURL(baseUrl, "/find-path-3d"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    await throwAPIError(response);
  }

  return response.json();
}

/**
 * Fetch backend health status with timeout.
 *
 * @param {string} [baseUrl=BACKEND_BASE_URL]
 * @param {{timeoutMs?:number}} [options]
 * @returns {Promise<{status:string, [key:string]:any}>}
 */
export async function fetchBackendHealth(baseUrl = BACKEND_BASE_URL, options = {}) {
  const timeoutMs = Number(options.timeoutMs || 4500);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(buildURL(baseUrl, "/health"), {
      method: "GET",
      signal: controller.signal,
    });
    if (!response.ok) {
      await throwAPIError(response);
    }
    return response.json();
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error("Backend health check timed out.");
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}
