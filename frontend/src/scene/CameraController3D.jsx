import { useEffect, useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

/**
 * CameraController3D handles multi-floor follow/orbit camera behavior.
 *
 * Modes:
 * - "orbit": user-controlled camera with OrbitControls
 * - "follow": smooth chase camera anchored to moving agent
 *
 * @param {object} props
 * @param {"orbit"|"follow"} props.mode
 * @param {{x:number,y:number,z:number}|null} props.agentPosition
 * @param {number} [props.agentYaw=0]
 * @param {number} [props.activeFloor=0]
 * @param {number} [props.floorHeightM=3.2]
 * @param {number} [props.followDistance=4.0]
 * @param {number} [props.followHeight=2.4]
 * @param {number} [props.smoothness=0.08]
 */
export default function CameraController3D({
  mode = "orbit",
  agentPosition,
  agentYaw = 0,
  activeFloor = 0,
  floorHeightM = 3.2,
  followDistance = 4.0,
  followHeight = 2.4,
  smoothness = 0.08,
}) {
  const { camera } = useThree();
  const controlsRef = useRef(null);
  const target = useMemo(() => new THREE.Vector3(), []);
  const desiredPos = useMemo(() => new THREE.Vector3(), []);
  const smoothLook = useMemo(() => new THREE.Vector3(), []);
  const initializedFollow = useRef(false);

  useEffect(() => {
    if (mode !== "orbit") return;
    initializedFollow.current = false;

    if (!controlsRef.current) return;
    if (agentPosition) {
      controlsRef.current.target.set(agentPosition.x, agentPosition.y ?? 0, agentPosition.z);
    } else {
      const baseY = activeFloor * floorHeightM;
      controlsRef.current.target.set(0, baseY, 0);
    }
    controlsRef.current.update();
  }, [mode, agentPosition, activeFloor, floorHeightM]);

  useFrame((_, delta) => {
    if (mode !== "follow") return;

    const baseY = activeFloor * floorHeightM;
    const anchorX = agentPosition?.x ?? 0;
    const anchorY = agentPosition?.y ?? baseY;
    const anchorZ = agentPosition?.z ?? 0;

    // Follow from behind agent heading while keeping full body and floor context in view.
    const bx = -Math.cos(agentYaw) * followDistance;
    const bz = -Math.sin(agentYaw) * followDistance;
    desiredPos.set(anchorX + bx, anchorY + followHeight, anchorZ + bz);
    target.set(anchorX, anchorY + 0.35, anchorZ);

    const positionLerp = 1 - Math.exp(-delta * 6.8);
    const lookLerp = 1 - Math.exp(-delta * 9.0);

    if (!initializedFollow.current) {
      camera.position.copy(desiredPos);
      smoothLook.copy(target);
      initializedFollow.current = true;
    } else {
      camera.position.lerp(desiredPos, positionLerp);
      smoothLook.lerp(target, lookLerp);
    }

    camera.lookAt(smoothLook);
  });

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enabled={mode === "orbit"}
      enableDamping
      dampingFactor={smoothness}
      minDistance={0.4}
      maxDistance={240}
      minPolarAngle={0.04}
      maxPolarAngle={Math.PI / 2 - 0.02}
    />
  );
}
