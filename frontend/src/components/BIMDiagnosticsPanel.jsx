import { useMemo } from "react";

/**
 * BIM diagnostics panel for scene graph + geometry validation results.
 *
 * @param {object} props
 * @param {object|null} props.projectMeta
 */
export default function BIMDiagnosticsPanel({ projectMeta }) {
  const report = projectMeta?.validation_report || null;
  const sceneGraph = projectMeta?.scene_graph || null;

  const issueGroups = useMemo(() => {
    const groups = {
      wall_intersection: 0,
      door_clearance: 0,
      wall_invalid: 0,
      other: 0,
    };

    for (const issue of report?.issues || []) {
      const kind = String(issue?.kind || "other");
      if (kind in groups) groups[kind] += 1;
      else groups.other += 1;
    }

    return groups;
  }, [report]);

  if (!projectMeta || projectMeta.mode !== "multi") {
    return (
      <section className="panel bim-panel" aria-label="BIM diagnostics">
        <h2>BIM Diagnostics</h2>
        <p className="muted">Process a multi-floor building to generate scene graph diagnostics.</p>
      </section>
    );
  }

  if (!report || !sceneGraph) {
    return (
      <section className="panel bim-panel" aria-label="BIM diagnostics">
        <h2>BIM Diagnostics</h2>
        <p className="muted">Scene graph diagnostics are not available for this build yet.</p>
      </section>
    );
  }

  const summary = report.summary || {};
  const warningCount = Number(summary.warnings || 0);
  const errorCount = Number(summary.errors || 0);
  const health = errorCount > 0 ? "Attention" : warningCount > 0 ? "Review" : "Healthy";

  return (
    <section className="panel bim-panel" aria-label="BIM diagnostics">
      <div className="bim-head">
        <h2>BIM Diagnostics</h2>
        <span className={`status-pill ${health === "Healthy" ? "is-ready" : health === "Review" ? "is-busy" : ""}`}>
          {health}
        </span>
      </div>

      <div className="kv">
        <span>Schema Version</span>
        <code>{sceneGraph.schema_version || "n/a"}</code>
      </div>
      <div className="kv">
        <span>Levels</span>
        <code>{summary.levels ?? sceneGraph.levels?.length ?? 0}</code>
      </div>
      <div className="kv">
        <span>Wall Checks</span>
        <code>{summary.wall_checks ?? 0}</code>
      </div>
      <div className="kv">
        <span>Door Checks</span>
        <code>{summary.door_checks ?? 0}</code>
      </div>
      <div className="kv">
        <span>Warnings / Errors</span>
        <code>
          {warningCount} / {errorCount}
        </code>
      </div>

      <div className="bim-issues-grid">
        <div className="bim-issue-chip">Wall Intersections: {issueGroups.wall_intersection}</div>
        <div className="bim-issue-chip">Door Clearance: {issueGroups.door_clearance}</div>
        <div className="bim-issue-chip">Invalid Walls: {issueGroups.wall_invalid}</div>
        <div className="bim-issue-chip">Other: {issueGroups.other}</div>
      </div>

      {Array.isArray(report.issues) && report.issues.length ? (
        <details className="bim-issues-list">
          <summary>Show issue details ({report.issues.length})</summary>
          <div className="bim-issues-items">
            {report.issues.slice(0, 20).map((issue, idx) => (
              <article key={`${issue.kind}-${idx}`} className="bim-issue-row">
                <strong>{String(issue.kind || "issue")}</strong>
                <span className="muted">{String(issue.message || "No message")}</span>
              </article>
            ))}
          </div>
        </details>
      ) : (
        <p className="muted">No geometry issues detected in current build.</p>
      )}
    </section>
  );
}
