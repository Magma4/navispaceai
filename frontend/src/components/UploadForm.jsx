import { useEffect, useState } from "react";

import { BACKEND_BASE_URL, processBlueprint, processBuilding } from "../api/backendAPI";

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
  const [files, setFiles] = useState([]);
  const [previewUrl, setPreviewUrl] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!files.length) {
      setPreviewUrl("");
      return undefined;
    }

    const objectUrl = URL.createObjectURL(files[0]);
    setPreviewUrl(objectUrl);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [files]);

  /**
   * Handle native file input changes.
   * @param {React.ChangeEvent<HTMLInputElement>} event
   */
  function handleFileChange(event) {
    const selected = Array.from(event.target.files || []);
    setFiles(selected);
  }

  /**
   * Process selected blueprint by calling POST /process-blueprint.
   */
  async function handleProcessClick() {
    if (!files.length || loading) return;

    setLoading(true);
    onProcessingChange?.(true);

    try {
      let result;
      if (files.length > 1) {
        result = await processBuilding(
          files,
          {
            floorNumbers: files.map((_, idx) => idx),
          },
          backendBaseUrl
        );
      } else {
        result = await processBlueprint(files[0], backendBaseUrl);
      }
      onProcessed?.(result);
      onNotify?.(
        files.length > 1
          ? `Processed ${files.length} floors successfully.`
          : "Blueprint processed successfully.",
        "success"
      );
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
      <p className="muted">Upload a clean PNG/JPG blueprint to generate a navigable 3D model.</p>

      <label className="file-input" htmlFor="blueprint-file">
        <span>Blueprint Image</span>
        <input
          id="blueprint-file"
          type="file"
          accept="image/png,image/jpeg,image/jpg"
          multiple
          onChange={handleFileChange}
          disabled={loading}
          aria-label="Select blueprint image"
        />
      </label>

      <div className="file-meta" aria-live="polite">
        {files.length ? (
          <>
            <strong>{files.length > 1 ? `${files.length} files selected` : files[0].name}</strong>
            <span>
              {(files.reduce((sum, file) => sum + file.size, 0) / 1024).toFixed(1)} KB total
            </span>
            {previewUrl ? (
              <div className="preview-wrap">
                <img src={previewUrl} alt="Blueprint preview" className="preview-image" />
              </div>
            ) : null}
          </>
        ) : (
          <span>No file selected yet.</span>
        )}
      </div>

      <button
        type="button"
        className="btn btn-primary"
        onClick={handleProcessClick}
        disabled={!files.length || loading}
        aria-disabled={!files.length || loading}
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
