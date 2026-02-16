import { useEffect } from "react";

/**
 * ToastNotification displays transient success/error messages.
 *
 * @param {object} props
 * @param {string|null} props.message - Toast message.
 * @param {"error"|"success"|"info"} [props.type] - Visual toast variant.
 * @param {() => void} props.onClose - Dismiss callback.
 * @param {number} [props.duration=3800] - Auto-dismiss delay in milliseconds.
 */
export default function ToastNotification({ message, type = "info", onClose, duration = 3800 }) {
  useEffect(() => {
    if (!message) return undefined;

    const timer = setTimeout(() => {
      onClose();
    }, duration);

    return () => clearTimeout(timer);
  }, [message, duration, onClose]);

  if (!message) return null;

  return (
    <div className={`toast toast-${type}`} role="status" aria-live="polite">
      <span>{message}</span>
      <button className="icon-btn" type="button" onClick={onClose} aria-label="Close notification">
        ×
      </button>
    </div>
  );
}
