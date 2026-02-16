import { useState } from "react";

import { BACKEND_BASE_URL, processBlueprint } from "../api/backendAPI";

/**
 * UploadForm handles blueprint file selection and processing API integration.
 *
 * @param {object} props
 * @param {string} [props.backendBaseUrl] - FastAPI backend base URL.
 * @param {(result:object)=>void} props.onProcessed - Called with backend process result.
 * @param {(message:string,type?:"error"|"success"|"info")=>void} [props.onNotify] - Optional toast callback.
 * @param {(loading:boolean)=>void} [props.onProcessingChange] - Optional loading state callback.
 */
export default function UploadForm({
  backendBaseUrl = BACKEND_BASE_URL,
  onProcessed,
  onNotify,
  onProcessingChange,
}) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  /**
   * Handle native file input changes.
   * @param {React.ChangeEvent<HTMLInputElement>} event
   */
  function handleFileChange(event) {
    const selected = event.target.files?.[0] || null;
    setFile(selected);
  }

  /**
   * Process selected blueprint by calling POST /process-blueprint.
   */
  async function handleProcessClick() {
    if (!file || loading) return;

    setLoading(true);
    onProcessingChange?.(true);

    try {
      const result = await processBlueprint(file, backendBaseUrl);
      onProcessed?.(result);
      onNotify?.("Blueprint processed successfully.", "success");
    } catch (error) {
      onNotify?.(error.message || "Blueprint processing failed.", "error");
    } finally {
      setLoading(false);
      onProcessingChange?.(false);
    }
  }

  return (
    <section className="panel upload-panel" aria-label="Blueprint upload panel">
      <h2>Upload Blueprint</h2>
      <p className="muted">PNG/JPG floor plan files are supported.</p>

      <label className="file-input" htmlFor="blueprint-file">
        <span>Blueprint Image</span>
        <input
          id="blueprint-file"
          type="file"
          accept="image/png,image/jpeg,image/jpg"
          onChange={handleFileChange}
          disabled={loading}
          aria-label="Select blueprint image"
        />
      </label>

      <div className="file-meta" aria-live="polite">
        {file ? (
          <>
            <strong>{file.name}</strong>
            <span>{(file.size / 1024).toFixed(1)} KB</span>
          </>
        ) : (
          <span>No file selected.</span>
        )}
      </div>

      <button
        type="button"
        className="btn btn-primary"
        onClick={handleProcessClick}
        disabled={!file || loading}
        aria-disabled={!file || loading}
      >
        {loading ? (
          <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
            <span className="spinner" aria-hidden="true" />
            Processing...
          </span>
        ) : (
          "Process Blueprint"
        )}
      </button>
    </section>
  );
}
