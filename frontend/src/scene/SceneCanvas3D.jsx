import { Suspense, useEffect, useMemo, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { Grid, Html, useGLTF } from "@react-three/drei";
import * as THREE from "three";

import CameraController3D from "./CameraController3D";

/**
 * FloorModel loads a floor-specific GLB and offsets it by floor elevation.
 *
 * @param {object} props
 * @param {string} props.url
 * @param {number} props.elevationM
 * @param {boolean} props.highlight
 */
function FloorModel({ url, elevationM, highlight }) {
  const { scene } = useGLTF(url);
  useEffect(() => {
    const wallMaterial = new THREE.MeshBasicMaterial({
      color: "#66758b",
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    const floorMaterial = new THREE.MeshBasicMaterial({
      color: "#dce8f5",
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    const doorMaterial = new THREE.MeshBasicMaterial({
      color: "#c58d63",
      side: THREE.DoubleSide,
      toneMapped: false,
    });

    /**
     * Infer mesh kind when exported names are absent.
     * @param {THREE.Mesh} mesh
     * @returns {"floor"|"door"|"wall"}
     */
    const inferKind = (mesh) => {
      const name = String(mesh.name || "").toLowerCase();
      if (name.includes("door")) return "door";
      if (name.includes("floor")) return "floor";
      if (name.includes("wall")) return "wall";

      const geometry = mesh.geometry;
      if (!geometry) return "wall";
      if (!geometry.boundingBox) geometry.computeBoundingBox();
      if (!geometry.boundingBox) return "wall";

      const size = new THREE.Vector3();
      geometry.boundingBox.getSize(size);
      const sy = Math.abs(size.y * mesh.scale.y);
      const sx = Math.abs(size.x * mesh.scale.x);
      const sz = Math.abs(size.z * mesh.scale.z);
      if (sy < 0.18 && Math.max(sx, sz) > 1.25) return "floor";
      if (sy > 1.3 && sy < 2.6 && Math.max(sx, sz) < 2.2) return "door";
      return "wall";
    };

    scene.traverse((obj) => {
      if (!obj.isMesh) return;
      obj.castShadow = false;
      obj.receiveShadow = false;
      const kind = inferKind(obj);
      if (kind === "floor") obj.material = floorMaterial;
      else if (kind === "door") obj.material = doorMaterial;
      else obj.material = wallMaterial;
    });

    return () => {
      wallMaterial.dispose();
      floorMaterial.dispose();
      doorMaterial.dispose();
    };
  }, [scene]);

  return (
    <group position={[0, elevationM, 0]}>
      <primitive object={scene} />
      {highlight ? (
        <mesh position={[0, 0.02, 0]}>
          <boxGeometry args={[50, 0.02, 50]} />
          <meshBasicMaterial color="#22c55e" transparent opacity={0.08} />
        </mesh>
      ) : null}
    </group>
  );
}

/**
 * Render room boundary line strips for indexed spaces.
 *
 * @param {object} props
 * @param {Array<object>} props.rooms
 * @param {number} props.activeFloor
 */
function RoomBoundaries({ rooms, activeFloor }) {
  if (!Array.isArray(rooms)) return null;

  return (
    <group>
      {rooms
        .filter((room) => Number(room.floor_number) === Number(activeFloor))
        .map((room) => {
          const points = room.polygon_m || [];
          if (points.length < 3) return null;

          return (
            <group key={room.room_id}>
              {points.map((point, index) => {
                const next = points[(index + 1) % points.length];
                const midX = (point[0] + next[0]) / 2;
                const midZ = (point[1] + next[1]) / 2;
                const len = Math.hypot(next[0] - point[0], next[1] - point[1]);
                const yaw = Math.atan2(next[1] - point[1], next[0] - point[0]);

                return (
                  <mesh
                    key={`${room.room_id}-${index}`}
                    position={[midX, room.centroid_m?.[1] ?? 0.02, midZ]}
                    rotation={[0, -yaw, 0]}
                  >
                    <boxGeometry args={[len, 0.03, 0.03]} />
                    <meshBasicMaterial color="#f59e0b" />
                  </mesh>
                );
              })}

              <Html position={[room.centroid_m?.[0] ?? 0, (room.centroid_m?.[1] ?? 0) + 0.2, room.centroid_m?.[2] ?? 0]}>
                <div style={{ fontSize: 10, background: "rgba(0,0,0,0.55)", color: "white", padding: "2px 6px", borderRadius: 6 }}>
                  {room.name}
                </div>
              </Html>
            </group>
          );
        })}
    </group>
  );
}

/**
 * Render 3D path with vertical transitions.
 *
 * @param {object} props
 * @param {Array<{x:number,y:number,z:number}>} props.path
 */
function Path3D({ path }) {
  const points = useMemo(() => {
    if (!Array.isArray(path) || path.length < 2) return [];
    return path.map((point) => new THREE.Vector3(point.x, point.y + 0.08, point.z));
  }, [path]);
  const geometry = useMemo(() => {
    if (points.length < 2) return null;
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [points]);

  useEffect(() => {
    return () => {
      geometry?.dispose();
    };
  }, [geometry]);

  if (points.length < 2 || !geometry) return null;

  return (
    <group>
      <line geometry={geometry}>
        <lineBasicMaterial color="#3b82f6" linewidth={3} />
      </line>
      {points.map((point, idx) => (
        <mesh key={`p-${idx}`} position={point}>
          <sphereGeometry args={[0.06, 12, 12]} />
          <meshBasicMaterial color={idx === 0 ? "#16a34a" : idx === points.length - 1 ? "#dc2626" : "#7ab4ff"} />
        </mesh>
      ))}
    </group>
  );
}

/**
 * Render animated navigation agent in 3D multi-floor space.
 *
 * @param {object} props
 * @param {{x:number,y:number,z:number}|null} props.position
 * @param {number} props.yaw
 */
function Agent3D({ position, yaw }) {
  if (!position) return null;

  return (
    <mesh position={[position.x, position.y + 0.2, position.z]} rotation={[0, -yaw, 0]} castShadow>
      <capsuleGeometry args={[0.14, 0.25, 8, 16]} />
      <meshStandardMaterial color="#f97316" />
    </mesh>
  );
}

/**
 * SceneCanvas3D renders full multi-floor building context.
 *
 * @param {object} props
 * @param {Array<object>} props.floors - Floor metadata from backend /floors
 * @param {Array<object>} props.rooms - Room metadata from backend /rooms
 * @param {number} props.activeFloor
 * @param {Array<{x:number,y:number,z:number}>} props.path
 * @param {{x:number,y:number,z:number}|null} props.agentPosition
 * @param {number} props.agentYaw
 * @param {"orbit"|"follow"} props.cameraMode
 * @param {(payload:{world:{x:number,y:number,z:number}})=>void} [props.onWorldPick]
 */
export default function SceneCanvas3D({
  floors,
  rooms,
  activeFloor,
  path,
  agentPosition,
  agentYaw,
  cameraMode,
  onWorldPick,
}) {
  const orderedFloors = useMemo(
    () => [...(Array.isArray(floors) ? floors : [])].sort((a, b) => Number(a.floor_number) - Number(b.floor_number)),
    [floors]
  );

  const floorHeightM = useMemo(() => {
    if (orderedFloors.length < 2) return 3.2;
    return Math.abs((orderedFloors[1].elevation_m ?? 3.2) - (orderedFloors[0].elevation_m ?? 0));
  }, [orderedFloors]);

  const centerRef = useRef([12, 8, 12]);
  const pickingPlaneY = useMemo(() => {
    const current = orderedFloors.find((floor) => Number(floor.floor_number) === Number(activeFloor));
    return Number(current?.elevation_m ?? 0);
  }, [activeFloor, orderedFloors]);

  /**
   * Handle click picking in world-space on active floor plane.
   * @param {import('@react-three/fiber').ThreeEvent<MouseEvent>} event
   */
  function handlePick(event) {
    event.stopPropagation();
    onWorldPick?.({
      world: {
        x: event.point.x,
        y: pickingPlaneY,
        z: event.point.z,
      },
    });
  }

  return (
    <div className="scene-shell" aria-label="Multi-floor 3D scene">
      <Canvas
        camera={{ position: centerRef.current, fov: 58, near: 0.1, far: 320 }}
        gl={{ antialias: true, logarithmicDepthBuffer: true }}
        shadows
      >
        <color attach="background" args={["#e7eef5"]} />
        <fog attach="fog" args={["#e7eef5", 20, 220]} />

        <ambientLight intensity={0.84} />
        <hemisphereLight intensity={0.45} color="#ffffff" groundColor="#95a8be" />
        <directionalLight position={[12, 22, 8]} intensity={1.0} castShadow />
        <directionalLight position={[-6, 14, -4]} intensity={0.34} />

        {/* Base world grid for orientation in meter units */}
        <Grid
          args={[250, 250]}
          cellSize={1}
          cellThickness={0.6}
          sectionSize={10}
          sectionThickness={1.1}
          fadeDistance={140}
          fadeStrength={1}
          position={[0, -0.05, 0]}
        />

        <mesh
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, pickingPlaneY, 0]}
          onClick={handlePick}
          visible={false}
        >
          <planeGeometry args={[500, 500]} />
          <meshBasicMaterial transparent opacity={0} />
        </mesh>

        {/* Render all floors stacked at real elevation */}
        {orderedFloors.map((floor) => (
          <group key={`floor-${floor.floor_number}`}>
            <Suspense fallback={null}>
              {floor.model_absolute_url || floor.model_url ? (
                <FloorModel
                  url={floor.model_absolute_url || floor.model_url}
                  elevationM={floor.elevation_m ?? floor.floor_number * floorHeightM}
                  highlight={Number(floor.floor_number) === Number(activeFloor)}
                />
              ) : null}
            </Suspense>
          </group>
        ))}

        <RoomBoundaries rooms={rooms} activeFloor={activeFloor} />
        <Path3D path={path} />
        <Agent3D position={agentPosition} yaw={agentYaw} />

        <CameraController3D
          mode={cameraMode}
          agentPosition={agentPosition}
          agentYaw={agentYaw}
          activeFloor={activeFloor}
          floorHeightM={floorHeightM}
        />
      </Canvas>
    </div>
  );
}
