import { createRoot } from "react-dom/client";

import GameApp from "./GameApp";
import "./styles/main.css";

/**
 * React entrypoint for standalone game tab (/game.html).
 */
const rootElement = document.getElementById("root");
const root = createRoot(rootElement);

root.render(<GameApp />);
