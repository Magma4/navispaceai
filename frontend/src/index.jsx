import React from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import "./styles/main.css";

/**
 * React entrypoint for NavispaceAI frontend.
 */
const rootElement = document.getElementById("root");
const root = createRoot(rootElement);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
