/**
 * Centralized FastAPI client for NavispaceAI frontend.
 *
 * Endpoints:
 * - POST /process-blueprint
 * - POST /find-path
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
