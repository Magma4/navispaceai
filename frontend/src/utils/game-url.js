/**
 * Utilities for generating and parsing shareable NavispaceAI game URLs.
 */

/**
 * Build a shareable game URL for /game.html.
 *
 * @param {object} params
 * @param {string} params.backendBaseUrl
 * @param {string} [params.modelUrl]
 * @param {string} [params.gridUrl]
 * @param {number} [params.cellSizeM]
 * @param {"single"|"multi"} [params.mode="single"]
 * @param {string} [params.origin]
 * @returns {string}
 */
export function buildGameURL({
  backendBaseUrl,
  modelUrl,
  gridUrl,
  cellSizeM,
  mode = "single",
  origin,
}) {
  const baseOrigin = origin || window.location.origin;
  const url = new URL("/game.html", baseOrigin);
  url.searchParams.set("backend", String(backendBaseUrl || "").trim());
  url.searchParams.set("mode", mode === "multi" ? "multi" : "single");
  if (modelUrl) url.searchParams.set("model", String(modelUrl || "").trim());
  if (gridUrl) url.searchParams.set("grid", String(gridUrl || "").trim());
  if (cellSizeM) url.searchParams.set("cell", String(cellSizeM || 0.2));
  return url.toString();
}

/**
 * Parse a game URL and validate required params.
 *
 * @param {string} href
 * @returns {{
 *   backendBaseUrl:string,
 *   modelUrl:string,
 *   gridUrl:string,
 *   cellSizeM:number,
 *   mode:"single"|"multi"
 * } | {error:string}}
 */
export function parseGameURL(href) {
  try {
    const url = new URL(href);
    const backendBaseUrl = url.searchParams.get("backend") || "";
    const modelUrl = url.searchParams.get("model") || "";
    const gridUrl = url.searchParams.get("grid") || "";
    const mode = url.searchParams.get("mode") === "multi" ? "multi" : "single";
    const cellSizeRaw = Number(url.searchParams.get("cell") || "0.2");
    const cellSizeM = Number.isFinite(cellSizeRaw) && cellSizeRaw > 0 ? cellSizeRaw : 0.2;

    if (!backendBaseUrl) {
      return {
        error: "Missing backend URL in game configuration.",
      };
    }

    if (mode === "single" && (!modelUrl || !gridUrl)) {
      return {
        error:
          "Missing single-floor game configuration in URL. Re-open the game from the upload dashboard so model/grid links are included.",
      };
    }

    return { backendBaseUrl, modelUrl, gridUrl, cellSizeM, mode };
  } catch {
    return { error: "Invalid game URL." };
  }
}
