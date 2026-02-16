import { useMemo } from "react";
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
  const target = useMemo(() => new THREE.Vector3(), []);
  const desiredPos = useMemo(() => new THREE.Vector3(), []);

  useFrame(() => {
    const baseY = activeFloor * floorHeightM;

    if (mode === "follow" && agentPosition) {
      // Follow from behind agent heading while keeping full body and floor context in view.
      const bx = -Math.cos(agentYaw) * followDistance;
      const bz = -Math.sin(agentYaw) * followDistance;
      desiredPos.set(agentPosition.x + bx, agentPosition.y + followHeight, agentPosition.z + bz);
      target.set(agentPosition.x, agentPosition.y + 0.35, agentPosition.z);
    } else {
      // Orbit mode default anchor is active floor centerline height.
      target.set(target.x, baseY, target.z);
      desiredPos.set(camera.position.x, Math.max(camera.position.y, baseY + 2.0), camera.position.z);
    }

    camera.position.lerp(desiredPos, smoothness);
    camera.lookAt(target);
  });

  return <OrbitControls makeDefault enabled={mode === "orbit"} enableDamping dampingFactor={0.08} />;
}
