import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

/**
 * PathRenderer visualizes navigation trajectory as a glowing tube.
 *
 * Features:
 * - Real-time updates when `path` changes
 * - Pulse animation for readable path directionality
 * - Optional debug points for collision probes / waypoints
 *
 * @param {object} props
 * @param {Array<{x:number,z:number,y?:number}>} props.path
 * @param {boolean} [props.debug=false]
 * @param {Array<{x:number,z:number,y?:number}>} [props.debugPoints=[]]
 */
export default function PathRenderer({ path, debug = false, debugPoints = [] }) {
  const glowMaterialRef = useRef(null);
  const flowHeadRef = useRef(null);
  const flowHaloRef = useRef(null);

  const geometryData = useMemo(() => {
    if (!Array.isArray(path) || path.length < 2) {
      return { curve: null, points: [] };
    }

    const points = path.map((p) => new THREE.Vector3(p.x, p.y ?? 0.12, p.z));
    const curve = new THREE.CatmullRomCurve3(points, false, "centripetal", 0.25);
    return { curve, points };
  }, [path]);

  useFrame(({ clock }) => {
    if (!glowMaterialRef.current) return;
    const pulse = 0.35 + Math.sin(clock.elapsedTime * 3.2) * 0.12;
    glowMaterialRef.current.opacity = Math.max(0.1, pulse);

    if (!geometryData.curve || !flowHeadRef.current || !flowHaloRef.current) return;

    const t = (clock.elapsedTime * 0.16) % 1;
    const p = geometryData.curve.getPointAt(t);
    flowHeadRef.current.position.copy(p);
    flowHaloRef.current.position.copy(p);

    const headPulse = 0.85 + Math.sin(clock.elapsedTime * 8.0) * 0.16;
    flowHeadRef.current.scale.setScalar(headPulse);
    flowHaloRef.current.scale.setScalar(0.95 + Math.sin(clock.elapsedTime * 4.0) * 0.1);
  });

  if (!geometryData.curve || geometryData.points.length < 2) return null;

  return (
    <group>
      {/* Core path tube */}
      <mesh userData={{ noCameraCollision: true }}>
        <tubeGeometry
          args={[geometryData.curve, Math.max(20, geometryData.points.length * 5), 0.055, 14, false]}
        />
        <meshStandardMaterial color="#2388ff" emissive="#1460d8" emissiveIntensity={0.8} />
      </mesh>

      {/* Soft glow overlay */}
      <mesh userData={{ noCameraCollision: true }}>
        <tubeGeometry
          args={[geometryData.curve, Math.max(20, geometryData.points.length * 5), 0.1, 12, false]}
        />
        <meshBasicMaterial ref={glowMaterialRef} color="#79beff" transparent opacity={0.3} depthWrite={false} />
      </mesh>

      {/* Moving waypoint head to communicate route direction and motion flow. */}
      <mesh ref={flowHaloRef} userData={{ noCameraCollision: true }}>
        <sphereGeometry args={[0.14, 18, 18]} />
        <meshBasicMaterial color="#7dd3fc" transparent opacity={0.24} depthWrite={false} />
      </mesh>
      <mesh ref={flowHeadRef} userData={{ noCameraCollision: true }}>
        <sphereGeometry args={[0.07, 18, 18]} />
        <meshStandardMaterial color="#dbeafe" emissive="#60a5fa" emissiveIntensity={0.9} />
      </mesh>

      {/* Optional debug points (e.g., collision probes or occupancy highlights) */}
      {debug
        ? debugPoints.map((point, index) => (
            <mesh
              key={`debug-${index}`}
              position={[point.x, (point.y ?? 0.08) + 0.04, point.z]}
              userData={{ noCameraCollision: true }}
            >
              <boxGeometry args={[0.06, 0.06, 0.06]} />
              <meshBasicMaterial color="#ff3b30" />
            </mesh>
          ))
        : null}
    </group>
  );
}
