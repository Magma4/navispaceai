import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * Vite configuration for React frontend.
 */
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
  },
});
