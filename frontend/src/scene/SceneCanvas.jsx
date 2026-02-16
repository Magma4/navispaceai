import { Suspense, useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Bounds, Grid, OrbitControls, useGLTF } from "@react-three/drei";
import * as THREE from "three";

import PathRenderer from "./PathRenderer";

/**
 * Convert grid cell to world-space coordinate.
 *
 * @param {{row:number,col:number}} cell
 * @param {number} cellSizeM
 * @returns {{x:number,z:number}}
 */
export function gridToWorld(cell, cellSizeM) {
  return {
    x: cell.col * cellSizeM,
    z: cell.row * cellSizeM,
  };
}

/**
 * Convert world-space coordinate to clamped grid cell.
 *
 * @param {{x:number,z:number}} point
 * @param {number} cellSizeM
 * @param {{rows:number,cols:number}} gridShape
 * @returns {{row:number,col:number}}
 */
export function worldToGrid(point, cellSizeM, gridShape) {
  const row = Math.max(0, Math.min(gridShape.rows - 1, Math.round(point.z / cellSizeM)));
  const col = Math.max(0, Math.min(gridShape.cols - 1, Math.round(point.x / cellSizeM)));
  return { row, col };
}

/**
 * Load and render backend-generated glTF model.
 *
 * @param {object} props
 * @param {string} props.url
 */
function BuildingModel({ url }) {
  const { scene } = useGLTF(url);
  useEffect(() => {
    scene.traverse((obj) => {
      // Ensure backend-generated meshes are visible with physically-based lighting.
      if (obj.isMesh) {
        obj.castShadow = true;
        obj.receiveShadow = true;
      }
    });
  }, [scene]);
  return <primitive object={scene} />;
}

/**
 * Render marker for start/goal cell selection.
 *
 * @param {object} props
 * @param {{row:number,col:number}|null} props.cell
 * @param {number} props.cellSizeM
 * @param {string} props.color
 */
function CellMarker({ cell, cellSizeM, color }) {
  if (!cell) return null;

  const pos = gridToWorld(cell, cellSizeM);
  return (
    <mesh position={[pos.x, 0.16, pos.z]}>
      <sphereGeometry args={[0.14, 20, 20]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.3} />
    </mesh>
  );
}

/**
 * Render movable autonomous agent.
 *
 * @param {object} props
 * @param {{x:number,z:number}|null} props.position
 * @param {number} props.yaw
 */
function AgentMesh({ position, yaw }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current || !position) return;
    ref.current.position.set(position.x, 0.2, position.z);
    ref.current.rotation.y = yaw;
  }, [position, yaw]);

  if (!position) return null;

  return (
    <mesh ref={ref} castShadow>
      <boxGeometry args={[0.28, 0.28, 0.28]} />
      <meshStandardMaterial color="#f97316" />
    </mesh>
  );
}

/**
 * Optional camera follow behavior that tracks the agent.
 *
 * @param {object} props
 * @param {boolean} props.enabled
 * @param {{x:number,z:number}|null} props.position
 */
function FollowCamera({ enabled, position }) {
  const { camera } = useThree();
  const lookTarget = useMemo(() => new THREE.Vector3(), []);

  useFrame(() => {
    if (!enabled || !position) return;

    lookTarget.set(position.x, 0.2, position.z);
    const desired = new THREE.Vector3(position.x + 3.0, 2.4, position.z + 3.0);
    camera.position.lerp(desired, 0.06);
    camera.lookAt(lookTarget);
  });

  return null;
}

/**
 * SceneCanvas renders the 3D navigation environment and translates click picks into grid cells.
 *
 * @param {object} props
 * @param {string} props.modelUrl - Absolute model URL.
 * @param {number[][]|null} props.occupancyGrid
 * @param {number} props.cellSizeM
 * @param {{row:number,col:number}|null} props.startCell
 * @param {{row:number,col:number}|null} props.goalCell
 * @param {Array<{x:number,z:number}>} props.pathWorld
 * @param {{x:number,z:number}|null} props.agentPosition
 * @param {number} props.agentYaw
 * @param {boolean} props.followCamera
 * @param {(payload:{cell:{row:number,col:number},world:{x:number,z:number}})=>void} props.onCellPick
 */
export default function SceneCanvas({
  modelUrl,
  occupancyGrid,
  cellSizeM,
  startCell,
  goalCell,
  pathWorld,
  agentPosition,
  agentYaw,
  followCamera,
  onCellPick,
}) {
  const gridShape = useMemo(() => {
    if (!Array.isArray(occupancyGrid) || !occupancyGrid.length) return null;
    return { rows: occupancyGrid.length, cols: occupancyGrid[0]?.length || 0 };
  }, [occupancyGrid]);

  /**
   * Handle click on large floor plane and map to occupancy grid coordinates.
   * @param {import('@react-three/fiber').ThreeEvent<MouseEvent>} event
   */
  function handlePlaneClick(event) {
    if (!gridShape || !cellSizeM) return;

    event.stopPropagation();

    const world = { x: event.point.x, z: event.point.z };
    const cell = worldToGrid(world, cellSizeM, gridShape);
    onCellPick?.({ cell, world });
  }

  const planeCenter = useMemo(() => {
    if (!gridShape || !cellSizeM) return [50, 0, 50];
    const width = gridShape.cols * cellSizeM;
    const depth = gridShape.rows * cellSizeM;
    return [width / 2, 0, depth / 2];
  }, [gridShape, cellSizeM]);

  const planeSize = useMemo(() => {
    if (!gridShape || !cellSizeM) return [1000, 1000];
    return [Math.max(1, gridShape.cols * cellSizeM), Math.max(1, gridShape.rows * cellSizeM)];
  }, [gridShape, cellSizeM]);

  return (
    <div className="scene-shell" aria-label="3D navigation scene">
      <Canvas camera={{ position: [11, 9, 11], fov: 58 }} shadows>
        <color attach="background" args={["#d7e2ec"]} />

        <ambientLight intensity={0.82} />
        <directionalLight position={[8, 14, 6]} intensity={1.45} castShadow />
        <directionalLight position={[-6, 8, -4]} intensity={0.45} />

        {/* Base floor plane to increase depth contrast and model readability. */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.025, 0]} receiveShadow>
          <planeGeometry args={[500, 500]} />
          <meshStandardMaterial color="#d3dde6" />
        </mesh>

        <Grid
          args={[200, 200]}
          cellSize={1}
          cellThickness={0.45}
          sectionSize={10}
          sectionThickness={0.85}
          fadeDistance={100}
          fadeStrength={1}
          position={[0, 0, 0]}
        />

        <mesh rotation={[-Math.PI / 2, 0, 0]} position={planeCenter} onClick={handlePlaneClick} visible={false}>
          <planeGeometry args={planeSize} />
          <meshBasicMaterial transparent opacity={0} />
        </mesh>

        <Suspense fallback={null}>
          {modelUrl ? (
            <Bounds fit clip observe margin={1.25}>
              <BuildingModel url={modelUrl} />
            </Bounds>
          ) : null}
        </Suspense>

        <CellMarker cell={startCell} cellSizeM={cellSizeM} color="#22c55e" />
        <CellMarker cell={goalCell} cellSizeM={cellSizeM} color="#ef4444" />

        <PathRenderer path={pathWorld} />
        <AgentMesh position={agentPosition} yaw={agentYaw} />
        <FollowCamera enabled={followCamera} position={agentPosition} />

        <axesHelper args={[1.5]} />
        <OrbitControls
          makeDefault
          enableDamping
          dampingFactor={0.09}
          minDistance={2}
          maxDistance={140}
          minPolarAngle={0.08}
          maxPolarAngle={Math.PI / 2 - 0.02}
        />
      </Canvas>
    </div>
  );
}
