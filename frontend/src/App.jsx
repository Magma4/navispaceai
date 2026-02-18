import { useEffect, useMemo, useRef, useState } from "react";

import UploadForm from "./components/UploadForm";
import FrameHero from "./components/FrameHero";
import ToastNotification from "./components/ToastNotification";
import {
  BACKEND_BASE_URL,
  fetchBackendHealth,
  resolveBackendURL,
} from "./api/backendAPI";
import { buildGameURL } from "./utils/gameURL";

const BACKEND_URL_STORAGE_KEY = "navispace_public_backend_url_v1";
const RECENT_BUILDS_STORAGE_KEY = "navispace_recent_builds_v1";

/**
 * Parse recent build records from localStorage.
 *
 * @returns {Array<object>}
 */
function loadRecentBuilds() {
  try {
    const raw = localStorage.getItem(RECENT_BUILDS_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.slice(0, 6) : [];
  } catch {
    return [];
  }
}

/**
 * Validate HTTP(S) URL.
 *
 * @param {string} value
 * @returns {boolean}
 */
function isValidHttpUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

/**
 * Estimate output quality confidence from returned metadata.
 *
 * @param {object|null} meta
 * @returns {{
 *   score:number,
 *   label:string,
 *   checks:Array<{name:string,ok:boolean,detail:string}>
 * }}
 */
function estimateBuildQuality(meta) {
  if (!meta) {
    return {
      score: 0,
      label: "No Build",
      checks: [],
    };
  }

  const walls = Array.isArray(meta.walls) ? meta.walls.length : 0;
  const doors = Array.isArray(meta.doors) ? meta.doors.length : 0;
  const rows = Number(meta?.grid_shape?.rows || 0);
  const cols = Number(meta?.grid_shape?.cols || 0);

  if (meta.mode === "multi") {
    const floorCount = Number(meta.floor_count || meta.grid3d_shape?.floors || 0);
    const roomCount = Number(meta.room_count || 0);
    const score = Math.max(
      0,
      Math.min(100, 50 + Math.min(floorCount * 8, 24) + Math.min(roomCount, 26))
    );

    return {
      score,
      label: score >= 80 ? "Strong" : score >= 60 ? "Good" : "Review",
      checks: [
        {
          name: "Floor Coverage",
          ok: floorCount >= 2,
          detail: `${floorCount} floor(s) processed`,
        },
        {
          name: "Room Indexing",
          ok: roomCount >= 4,
          detail: `${roomCount} room(s) indexed`,
        },
      ],
    };
  }

  const densityBase = Math.max(1, rows + cols);
  const wallDensity = walls / densityBase;
  const doorRatio = doors / Math.max(1, walls);

  let score = 100;
  if (walls < 12) score -= 35;
  if (wallDensity < 0.12) score -= 20;
  if (doors < 2) score -= 14;
  if (doorRatio < 0.01) score -= 14;

  score = Math.max(0, Math.min(100, score));

  return {
    score,
    label: score >= 82 ? "Strong" : score >= 64 ? "Good" : "Review",
    checks: [
      {
        name: "Wall Detection",
        ok: walls >= 12,
        detail: `${walls} wall segment(s)`,
      },
      {
        name: "Door Detection",
        ok: doors >= 2,
        detail: `${doors} door candidate(s)`,
      },
      {
        name: "Grid Scale",
        ok: rows >= 32 && cols >= 32,
        detail: `${rows} × ${cols} cells`,
      },
    ],
  };
}

/**
 * Main client-facing dashboard:
 * - Cinematic landing hero
 * - Blueprint processing studio
 * - Shareable runtime publishing workflow
 */
export default function App() {
  const [processing, setProcessing] = useState(false);
  const [projectMeta, setProjectMeta] = useState(null);
  const [publicBackendUrl, setPublicBackendUrl] = useState(() => {
    const stored = localStorage.getItem(BACKEND_URL_STORAGE_KEY);
    return stored && isValidHttpUrl(stored) ? stored : BACKEND_BASE_URL;
  });
  const [recentBuilds, setRecentBuilds] = useState(() => loadRecentBuilds());
  const [backendHealth, setBackendHealth] = useState({
    status: "checking",
    detail: "Checking backend...",
    latencyMs: null,
    payload: null,
  });
  const [toast, setToast] = useState({ message: null, type: "info" });
  const [showStudioDetails, setShowStudioDetails] = useState(false);

  const recentSignatureRef = useRef("");

  /**
   * Show toast notification.
   *
   * @param {string} message
   * @param {"error"|"success"|"info"} type
   */
  function showToast(message, type = "info") {
    setToast({ message, type });
  }

  /** Handle successful backend processing payload from UploadForm. */
  function handleProcessed(result) {
    const isMulti = Array.isArray(result?.floors) && result.floors.length > 0;
    setProjectMeta({
      ...result,
      mode: isMulti ? "multi" : "single",
      processed_at: new Date().toISOString(),
    });
    showToast(
      isMulti
        ? "Building processed. Multi-floor runtime link is ready."
        : "Blueprint processed. Runtime link is ready.",
      "success"
    );

    if (!isMulti && result?.ml_enabled && !result?.ml_engine_loaded) {
      showToast(
        `ML enabled but not loaded (${result?.ml_reason || "unknown reason"}). Falling back to heuristic detection.`,
        "error"
      );
    } else if (!isMulti && !result?.ml_used) {
      showToast("Inference fell back to classical detection for this blueprint.", "info");
    }
  }

  const backendUrlValid = useMemo(
    () => isValidHttpUrl(publicBackendUrl),
    [publicBackendUrl]
  );

  const normalizedBackendUrl = useMemo(() => {
    if (!backendUrlValid) return "";
    return String(publicBackendUrl).replace(/\/$/, "");
  }, [backendUrlValid, publicBackendUrl]);

  const resolvedModelURL = useMemo(() => {
    if (!projectMeta || projectMeta.mode === "multi" || !backendUrlValid) return "";
    return resolveBackendURL(
      projectMeta.model_url || projectMeta.model_absolute_url,
      normalizedBackendUrl
    );
  }, [backendUrlValid, normalizedBackendUrl, projectMeta]);

  const resolvedGridURL = useMemo(() => {
    if (!projectMeta || projectMeta.mode === "multi" || !backendUrlValid) return "";
    return resolveBackendURL(projectMeta.grid_url, normalizedBackendUrl);
  }, [backendUrlValid, normalizedBackendUrl, projectMeta]);

  const gameUrl = useMemo(() => {
    if (!projectMeta || !backendUrlValid) return "";
    if (projectMeta.mode === "multi") {
      return buildGameURL({
        backendBaseUrl: normalizedBackendUrl,
        mode: "multi",
      });
    }
    if (!resolvedModelURL || !resolvedGridURL) return "";
    return buildGameURL({
      backendBaseUrl: normalizedBackendUrl,
      modelUrl: resolvedModelURL,
      gridUrl: resolvedGridURL,
      cellSizeM: projectMeta.cell_size_m || 0.2,
    });
  }, [backendUrlValid, normalizedBackendUrl, projectMeta, resolvedGridURL, resolvedModelURL]);

  const gridText = useMemo(() => {
    if (!projectMeta) return "Not loaded";
    if (projectMeta.mode === "multi") {
      const floors = Number(projectMeta.grid3d_shape?.floors || projectMeta.floor_count || 0);
      const rows = Number(projectMeta.grid3d_shape?.rows || 0);
      const cols = Number(projectMeta.grid3d_shape?.cols || 0);
      return floors ? `${floors}F · ${rows} × ${cols}` : `${projectMeta.floor_count || 0} floors`;
    }
    return `${projectMeta.grid_shape.rows} × ${projectMeta.grid_shape.cols}`;
  }, [projectMeta]);

  const quality = useMemo(() => estimateBuildQuality(projectMeta), [projectMeta]);

  /**
   * Save backend URL preference for future sessions.
   */
  useEffect(() => {
    if (!backendUrlValid) return;
    localStorage.setItem(BACKEND_URL_STORAGE_KEY, normalizedBackendUrl);
  }, [backendUrlValid, normalizedBackendUrl]);

  /**
   * Persist recent builds when a new project is processed.
   */
  useEffect(() => {
    if (!projectMeta || !gameUrl) return;

    const signature = `${projectMeta.processed_at || ""}|${projectMeta.mode}|${gameUrl}`;
    if (signature === recentSignatureRef.current) return;
    recentSignatureRef.current = signature;

    const record = {
      id: `${Date.now()}`,
      processedAt: projectMeta.processed_at || new Date().toISOString(),
      mode: projectMeta.mode,
      gridText,
      gameUrl,
      backendUrl: normalizedBackendUrl,
      modelUrl: resolvedModelURL || "",
      gridUrl: resolvedGridURL || "",
    };

    const next = [record, ...recentBuilds.filter((item) => item.gameUrl !== record.gameUrl)].slice(0, 6);
    setRecentBuilds(next);
    localStorage.setItem(RECENT_BUILDS_STORAGE_KEY, JSON.stringify(next));
  }, [
    gameUrl,
    gridText,
    normalizedBackendUrl,
    projectMeta,
    recentBuilds,
    resolvedGridURL,
    resolvedModelURL,
  ]);

  /**
   * Check backend health for client readiness status.
   */
  async function runHealthCheck() {
    if (!backendUrlValid) {
      setBackendHealth({
        status: "invalid",
        detail: "Enter a valid backend URL.",
        latencyMs: null,
        payload: null,
      });
      return;
    }

    setBackendHealth((prev) => ({
      ...prev,
      status: "checking",
      detail: "Checking backend...",
    }));

    const started = performance.now();
    try {
      const payload = await fetchBackendHealth(normalizedBackendUrl, { timeoutMs: 4500 });
      const latency = Math.round(performance.now() - started);
      setBackendHealth({
        status: "online",
        detail: payload?.status === "ok" ? "Backend healthy" : "Backend responded",
        latencyMs: latency,
        payload,
      });
    } catch (error) {
      setBackendHealth({
        status: "offline",
        detail: error.message || "Backend unavailable",
        latencyMs: null,
        payload: null,
      });
    }
  }

  /**
   * Auto-run health checks with debounce when backend URL changes.
   */
  useEffect(() => {
    const timer = setTimeout(() => {
      runHealthCheck();
    }, 280);
    return () => clearTimeout(timer);
  }, [normalizedBackendUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * Copy text to clipboard with unified toast handling.
   *
   * @param {string} text
   * @param {string} successMessage
   */
  async function copyText(text, successMessage) {
    if (!text) {
      showToast("Nothing to copy yet.", "error");
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      showToast(successMessage, "success");
    } catch {
      showToast("Clipboard access failed. Copy manually.", "error");
    }
  }

  /** Export a portable build manifest for client deployment handoff. */
  function downloadManifest() {
    if (!projectMeta || !gameUrl) {
      showToast("Process blueprint(s) first to export a manifest.", "error");
      return;
    }

    const manifest = {
      generated_at: new Date().toISOString(),
      product: "NavispaceAI",
      backend_url: normalizedBackendUrl,
      mode: projectMeta.mode,
      quality,
      assets: {
        game_url: gameUrl,
        model_url: resolvedModelURL || null,
        grid_url: resolvedGridURL || null,
      },
      metadata: {
        grid_shape: projectMeta.grid_shape || null,
        grid3d_shape: projectMeta.grid3d_shape || null,
        floor_count: projectMeta.floor_count || null,
        room_count: projectMeta.room_count || null,
        cell_size_m: projectMeta.cell_size_m || null,
      },
    };

    const blob = new Blob([JSON.stringify(manifest, null, 2)], {
      type: "application/json;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `navispace_manifest_${Date.now()}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
    showToast("Build manifest downloaded.", "success");
  }

  /** Open standalone game in a new tab. */
  function openGameInNewTab() {
    if (!gameUrl) {
      showToast("Process a blueprint first to generate a game URL.", "error");
      return;
    }
    window.open(gameUrl, "_blank", "noopener,noreferrer");
  }

  /**
   * Scroll to build studio section.
   */
  function scrollToStudio() {
    const el = document.getElementById("build-studio");
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  const backendBadgeClass =
    backendHealth.status === "online"
      ? "status-pill is-ready"
      : backendHealth.status === "checking"
        ? "status-pill is-busy"
        : "status-pill";

  const runtimeStatusText = gameUrl
    ? "Runtime link generated and ready to share."
    : "Process blueprint(s) to generate runtime link.";

  return (
    <div className="app dashboard-app dashboard-scroll">
      <div className="dashboard-bg" aria-hidden="true">
        <span className="orb orb-a" />
        <span className="orb orb-b" />
        <span className="orb orb-c" />
        <span className="grid-glow" />
      </div>

      <header className="topbar dashboard-topbar">
        <div className="brand-wrap dashboard-brand">
          <div className="brand-icon dashboard-brand-icon" aria-hidden="true">
            N
          </div>
          <div className="brand-copy">
            <h1>NavispaceAI</h1>
            <p>Enterprise Blueprint Intelligence Platform</p>
          </div>
        </div>

        <div className="topbar-metrics dashboard-metrics">
          <span className={`status-pill ${processing ? "is-busy" : "is-ready"}`}>
            {processing ? "Processing..." : "Studio Ready"}
          </span>
          <span className={backendBadgeClass}>
            {backendHealth.status === "online"
              ? `API Online${backendHealth.latencyMs ? ` · ${backendHealth.latencyMs}ms` : ""}`
              : backendHealth.status === "checking"
                ? "Checking API..."
                : "API Attention"}
          </span>
          <span className="status-pill">
            {projectMeta?.mode === "multi" ? "Multi-floor Runtime" : "Single-floor Runtime"}
          </span>
          <span className="backend-pill">Backend: {normalizedBackendUrl || "Invalid URL"}</span>
        </div>
      </header>

      <FrameHero onPrimaryAction={scrollToStudio} />

      <main className="landing-shell" id="build-studio">
        <section className="landing-primary panel">
          <div className="landing-card-head">
            <span className="landing-eyebrow">Build Studio</span>
            <h2>Create immersive indoor navigation runtimes</h2>
            <p>
              Upload blueprint assets, process to 3D, and publish a shareable game runtime for your
              team or clients.
            </p>
          </div>

          <UploadForm
            backendBaseUrl={normalizedBackendUrl || BACKEND_BASE_URL}
            onProcessed={handleProcessed}
            onNotify={showToast}
            onProcessingChange={setProcessing}
          />

          <section className="landing-quick-actions">
            <div className="action-row">
              <button type="button" className="btn btn-primary" onClick={openGameInNewTab} disabled={!gameUrl}>
                Open 3D Runtime
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => copyText(gameUrl, "Runtime URL copied.")}
                disabled={!gameUrl}
              >
                Copy Runtime URL
              </button>
              <button type="button" className="btn btn-ghost" onClick={downloadManifest} disabled={!gameUrl}>
                Export Manifest
              </button>
            </div>
            <p className="muted">{runtimeStatusText}</p>
          </section>
        </section>

        <aside className="landing-sidebar">
          <section className="panel landing-summary-card" aria-label="Executive summary">
            <span className="landing-eyebrow">Executive Summary</span>
            <h3>{projectMeta ? "Latest build is ready for review." : "Ready for your first build."}</h3>
            <p className="muted">
              A clean production pipeline from blueprint upload to navigable 3D runtime, with
              enterprise deployment handoff.
            </p>
            <div className="kv">
              <span>Processing Mode</span>
              <code>{projectMeta?.mode === "multi" ? "Multi-floor 3D" : "Single-floor 3D"}</code>
            </div>
            <div className="kv">
              <span>Grid</span>
              <code>{gridText}</code>
            </div>
            <div className="kv">
              <span>Build Quality</span>
              <code>{projectMeta ? `${quality.label} (${quality.score}/100)` : "Pending"}</code>
            </div>
          </section>

          <section className="panel studio-toggle-card" aria-label="Studio details toggle">
            <h3>Professional Studio Details</h3>
            <p className="muted">
              Advanced deployment and telemetry controls are available when needed.
            </p>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => setShowStudioDetails((prev) => !prev)}
            >
              {showStudioDetails ? "Hide Studio Details" : "Show Studio Details"}
            </button>
          </section>
        </aside>
      </main>

      {showStudioDetails ? (
        <section className="studio-details-grid" aria-label="Studio details">
          <section className="panel publish-panel" aria-label="Publishing configuration">
            <h2>Client Publishing</h2>
            <label className="field-group" htmlFor="public-backend-url">
              <span className="muted">Public Backend URL</span>
              <input
                id="public-backend-url"
                type="url"
                value={publicBackendUrl}
                onChange={(event) => setPublicBackendUrl(event.target.value)}
                placeholder="https://api.yourcompany.com"
              />
            </label>
            <label className="field-group" htmlFor="shareable-game-url">
              <span className="muted">Shareable Runtime URL</span>
              <input
                id="shareable-game-url"
                type="text"
                value={
                  backendUrlValid
                    ? gameUrl || "Process blueprint(s) to generate runtime URL."
                    : "Enter a valid http(s) backend URL first."
                }
                readOnly
              />
            </label>
            <div className="action-row" style={{ marginTop: 8 }}>
              <button type="button" className="btn btn-secondary" onClick={runHealthCheck}>
                Check API Health
              </button>
              <button
                type="button"
                className="btn btn-ghost"
                onClick={() => copyText(normalizedBackendUrl, "Backend URL copied.")}
                disabled={!backendUrlValid}
              >
                Copy API URL
              </button>
            </div>
          </section>

          <section className="panel health-panel" aria-label="Backend health">
            <h2>Runtime Health</h2>
            <p className="muted">{backendHealth.detail}</p>
            <div className="kv">
              <span>Status</span>
              <code>{backendHealth.status}</code>
            </div>
            <div className="kv">
              <span>Latency</span>
              <code>{backendHealth.latencyMs ? `${backendHealth.latencyMs} ms` : "n/a"}</code>
            </div>
            <div className="kv">
              <span>ML Inference</span>
              <code>
                {backendHealth.payload?.ml_enabled
                  ? backendHealth.payload?.ml_model_configured
                    ? "Enabled"
                    : "Enabled (model missing)"
                  : "Disabled"}
              </code>
            </div>
          </section>

          <section className="panel quality-panel" aria-label="Build quality checks">
            <h2>Build Quality Checks</h2>
            {!projectMeta ? (
              <p className="muted">Run processing to generate quality checks.</p>
            ) : (
              <>
                {quality.checks.map((item) => (
                  <div className="kv" key={item.name}>
                    <span>{item.name}</span>
                    <code>{item.ok ? `PASS · ${item.detail}` : `REVIEW · ${item.detail}`}</code>
                  </div>
                ))}
              </>
            )}
          </section>

          <section className="panel recent-panel" aria-label="Recent builds">
            <h2>Recent Builds</h2>
            {recentBuilds.length ? (
              <div className="recent-builds">
                {recentBuilds.map((item) => (
                  <article className="recent-build-card" key={item.id}>
                    <div className="recent-build-head">
                      <strong>{item.mode === "multi" ? "Multi-floor Build" : "Single-floor Build"}</strong>
                      <span>{new Date(item.processedAt).toLocaleString()}</span>
                    </div>
                    <p>{item.gridText}</p>
                    <div className="action-row">
                      <a className="btn btn-sm btn-primary" href={item.gameUrl} target="_blank" rel="noreferrer">
                        Open
                      </a>
                      <button
                        type="button"
                        className="btn btn-sm btn-secondary"
                        onClick={() => copyText(item.gameUrl, "Recent runtime URL copied.")}
                      >
                        Copy URL
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <p className="muted">No recent builds yet.</p>
            )}
          </section>

          <section className="panel deployment-panel" aria-label="Deployment handoff guide">
            <h2>Deployment Handoff</h2>
            <p className="muted">
              Production workflow for customer-ready deployments on your organization infrastructure.
            </p>
            <div className="kv">
              <span>1. Build frontend</span>
              <code>cd frontend && npm run build</code>
            </div>
            <div className="kv">
              <span>2. Host static files</span>
              <code>Deploy frontend/dist to your domain</code>
            </div>
            <div className="kv">
              <span>3. Expose backend API</span>
              <code>{normalizedBackendUrl || "https://api.yourcompany.com"}</code>
            </div>
            <div className="kv">
              <span>4. Share runtime URL</span>
              <code>{gameUrl || "(generated after processing)"}</code>
            </div>
          </section>
        </section>
      ) : null}

      <ToastNotification
        message={toast.message}
        type={toast.type}
        onClose={() => setToast({ message: null, type: "info" })}
      />
    </div>
  );
}
