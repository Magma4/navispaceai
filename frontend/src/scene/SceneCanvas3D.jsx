import { useMemo, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { Grid, Html, useGLTF } from "@react-three/drei";

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
  if (!Array.isArray(path) || path.length < 2) return null;

  return (
    <group>
      {path.slice(1).map((curr, i) => {
        const prev = path[i];
        const dx = curr.x - prev.x;
        const dy = curr.y - prev.y;
        const dz = curr.z - prev.z;
        const length = Math.hypot(dx, dy, dz);
        const mid = [prev.x + dx / 2, prev.y + dy / 2, prev.z + dz / 2];

        return (
          <mesh key={`seg-${i}`} position={mid}>
            <boxGeometry args={[0.06, 0.06, Math.max(0.08, length)]} />
            <meshStandardMaterial color="#3b82f6" emissive="#1d4ed8" emissiveIntensity={0.8} />
          </mesh>
        );
      })}
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
 */
export default function SceneCanvas3D({
  floors,
  rooms,
  activeFloor,
  path,
  agentPosition,
  agentYaw,
  cameraMode,
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

  return (
    <div className="scene-shell" aria-label="Multi-floor 3D scene">
      <Canvas camera={{ position: centerRef.current, fov: 58 }} shadows>
        <color attach="background" args={["#e7eef5"]} />

        <ambientLight intensity={0.62} />
        <directionalLight position={[12, 22, 8]} intensity={1.1} castShadow />

        {/* Base world grid for orientation in meter units */}
        <Grid
          args={[250, 250]}
          cellSize={1}
          cellThickness={0.6}
          sectionSize={10}
          sectionThickness={1.1}
          fadeDistance={140}
          fadeStrength={1}
          position={[0, 0, 0]}
        />

        {/* Render all floors stacked at real elevation */}
        {orderedFloors.map((floor) => (
          <group key={`floor-${floor.floor_number}`}>
            {floor.model_url ? (
              <FloorModel
                url={floor.model_url}
                elevationM={floor.elevation_m ?? floor.floor_number * floorHeightM}
                highlight={Number(floor.floor_number) === Number(activeFloor)}
              />
            ) : null}
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
