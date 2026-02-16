import { useMemo } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

/**
 * CameraController supports two modes:
 * - orbit: user-controlled free camera
 * - follow: smooth chase camera centered on moving agent
 *
 * @param {object} props
 * @param {"orbit"|"follow"} props.mode
 * @param {{x:number,z:number,y?:number}|null} props.agentPosition
 * @param {number} [props.agentYaw=0]
 * @param {number} [props.smoothness=0.08]
 * @param {number} [props.followDistance=3.5]
 * @param {number} [props.followHeight=2.6]
 */
export default function CameraController({
  mode = "orbit",
  agentPosition,
  agentYaw = 0,
  smoothness = 0.08,
  followDistance = 3.5,
  followHeight = 2.6,
}) {
  const { camera } = useThree();

  const lookTarget = useMemo(() => new THREE.Vector3(), []);
  const desiredPos = useMemo(() => new THREE.Vector3(), []);

  useFrame(() => {
    if (mode !== "follow" || !agentPosition) return;

    // Follow behind the agent orientation while keeping it centered and visible.
    const backwardX = -Math.cos(agentYaw) * followDistance;
    const backwardZ = -Math.sin(agentYaw) * followDistance;

    desiredPos.set(agentPosition.x + backwardX, (agentPosition.y || 0.2) + followHeight, agentPosition.z + backwardZ);
    lookTarget.set(agentPosition.x, (agentPosition.y || 0.2) + 0.35, agentPosition.z);

    camera.position.lerp(desiredPos, smoothness);
    camera.lookAt(lookTarget);
  });

  return <OrbitControls makeDefault enabled={mode === "orbit"} enableDamping dampingFactor={0.08} />;
}
