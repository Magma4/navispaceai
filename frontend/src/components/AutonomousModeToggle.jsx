/**
 * AutonomousModeToggle enables and controls autonomous navigation mode.
 *
 * @param {object} props
 * @param {boolean} props.enabled - Autonomous mode state.
 * @param {boolean} props.disabled - If true, toggle is disabled.
 * @param {(next:boolean)=>void} props.onToggle - Called when toggle changes.
 * @param {boolean} props.followCamera - Camera follow state.
 * @param {(next:boolean)=>void} props.onFollowCameraChange - Called when follow toggle changes.
 */
export default function AutonomousModeToggle({
  enabled,
  disabled,
  onToggle,
  followCamera,
  onFollowCameraChange,
}) {
  return (
    <section className="panel auto-panel" aria-label="Autonomous navigation controls">
      <h2>Autonomous Navigation</h2>
      <p className="muted">Enable hands-free movement along computed routes with optional chase camera.</p>

      <button
        className={`btn ${enabled ? "btn-danger" : "btn-primary"}`}
        type="button"
        onClick={() => onToggle(!enabled)}
        disabled={disabled}
        aria-pressed={enabled}
        aria-label="Toggle autonomous navigation mode"
      >
        {enabled ? "Disable Autonomous Mode" : "Enable Autonomous Mode"}
      </button>

      <label className="checkbox compact" htmlFor="follow-camera">
        <input
          id="follow-camera"
          type="checkbox"
          checked={followCamera}
          onChange={(event) => onFollowCameraChange(event.target.checked)}
          disabled={!enabled}
          aria-label="Follow agent with camera"
        />
        <span>Follow Camera</span>
      </label>
    </section>
  );
}
