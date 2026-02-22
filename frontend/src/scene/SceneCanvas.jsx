import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Grid, OrbitControls, useGLTF } from "@react-three/drei";
import * as THREE from "three";

import PathRenderer from "./PathRenderer";
import { isWorldPositionBlocked } from "./CollisionManager";

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
 * Build a lightweight procedural texture.
 *
 * @param {string} c1
 * @param {string} c2
 * @param {number} tile
 * @param {number} stripes
 * @returns {THREE.CanvasTexture}
 */
function makeStripedTexture(c1, c2, tile = 256, stripes = 10) {
  const canvas = document.createElement("canvas");
  canvas.width = tile;
  canvas.height = tile;
  const ctx = canvas.getContext("2d");

  if (!ctx) {
    const fallback = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
    fallback.needsUpdate = true;
    return fallback;
  }

  ctx.fillStyle = c1;
  ctx.fillRect(0, 0, tile, tile);

  ctx.strokeStyle = c2;
  ctx.lineWidth = Math.max(1, Math.floor(tile / (stripes * 7)));

  const step = tile / stripes;
  for (let i = -stripes; i < stripes * 2; i += 1) {
    ctx.beginPath();
    ctx.moveTo(i * step, 0);
    ctx.lineTo(i * step - tile, tile);
    ctx.stroke();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(12, 12);
  texture.anisotropy = 8;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Load and render backend-generated glTF model with richer materials.
 *
 * @param {object} props
 * @param {string} props.url
 * @param {(bounds:{center:THREE.Vector3,size:THREE.Vector3,radius:number})=>void} [props.onBounds]
 */
function BuildingModel({ url, onBounds }) {
  const { scene } = useGLTF(url);

  useEffect(() => {
    // Keep materials simple and high-contrast to avoid dark/black artifacts from malformed UVs/normals.
    const wallMaterial = new THREE.MeshLambertMaterial({
      color: "#8fa6bf",
      side: THREE.DoubleSide,
    });

    const floorMaterial = new THREE.MeshLambertMaterial({
      color: "#dbe6f2",
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.92,
    });

    const doorMaterial = new THREE.MeshStandardMaterial({
      color: "#ff7a1a",
      emissive: "#ff7a1a",
      emissiveIntensity: 0.28,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.95,
      roughness: 0.45,
      metalness: 0.06,
    });

    const stairMaterial = new THREE.MeshStandardMaterial({
      color: "#2da4ff",
      emissive: "#2da4ff",
      emissiveIntensity: 0.18,
      side: THREE.DoubleSide,
      roughness: 0.52,
      metalness: 0.04,
    });

    /**
     * Infer semantic type when mesh names are missing in exported GLB.
     * @param {THREE.Mesh} obj
     * @returns {"floor"|"door"|"stair"|"wall"|"unknown"}
     */
    const inferMeshType = (obj) => {
      const name = String(obj.name || "").toLowerCase();
      if (name.startsWith("door_") || name.includes("door")) return "door";
      if (name.startsWith("stair_") || name.includes("stair")) return "stair";
      if (name.includes("floor")) return "floor";
      if (name.startsWith("wall_") || name.includes("wall")) return "wall";

      const geom = obj.geometry;
      if (!geom) return "unknown";
      if (!geom.boundingBox) geom.computeBoundingBox();
      if (!geom.boundingBox) return "unknown";

      const size = new THREE.Vector3();
      geom.boundingBox.getSize(size);
      const sx = Math.abs(size.x * obj.scale.x);
      const sy = Math.abs(size.y * obj.scale.y);
      const sz = Math.abs(size.z * obj.scale.z);
      const longSide = Math.max(sx, sz);
      const shortSide = Math.min(sx, sz);

      if (sy < 0.2 && longSide > 1.4) return "floor";
      if (sy > 1.65 && sy < 2.5 && shortSide < 0.14 && longSide < 1.5) return "door";
      if (sy <= 2.0 && shortSide >= 0.2 && shortSide <= 1.4 && longSide >= 0.35 && longSide <= 3.0) return "stair";
      return "unknown";
    };

    scene.traverse((obj) => {
      if (!obj.isMesh) return;

      // Disable mesh self-shadowing: generated GLBs can otherwise render as black slabs.
      obj.castShadow = false;
      obj.receiveShadow = true;
      if (obj.geometry && !obj.geometry.attributes?.normal) {
        obj.geometry.computeVertexNormals();
      }

      const kind = inferMeshType(obj);
      if (kind === "door") obj.material = doorMaterial;
      else if (kind === "stair") obj.material = stairMaterial;
      else if (kind === "floor") obj.material = floorMaterial;
      else if (kind === "wall") obj.material = wallMaterial;
      // unknown: preserve original GLB material instead of flattening to walls
    });

    const box = new THREE.Box3().setFromObject(scene);
    if (!box.isEmpty()) {
      const center = new THREE.Vector3();
      const size = new THREE.Vector3();
      const sphere = new THREE.Sphere();
      box.getCenter(center);
      box.getSize(size);
      box.getBoundingSphere(sphere);
      onBounds?.({
        center: center.clone(),
        size: size.clone(),
        radius: Math.max(1, sphere.radius),
      });
    }

    return () => {
      wallMaterial.dispose();
      floorMaterial.dispose();
      doorMaterial.dispose();
      stairMaterial.dispose();
    };
  }, [scene, onBounds]);

  return <primitive object={scene} />;
}

/**
 * Command-driven camera transitions for fit and aerial views.
 *
 * @param {object} props
 * @param {{center:THREE.Vector3,size:THREE.Vector3,radius:number}|null} props.modelBounds
 * @param {boolean} props.followCamera
 * @param {boolean} props.immersiveMode
 * @param {React.MutableRefObject<any>} props.controlsRef
 * @param {{mode:"fit"|"aerial",nonce:number}|null} props.presetRequest
 */
function ViewController({ modelBounds, followCamera, immersiveMode, controlsRef, presetRequest }) {
  const { camera } = useThree();
  const startPos = useMemo(() => new THREE.Vector3(), []);
  const startTarget = useMemo(() => new THREE.Vector3(), []);
  const endPos = useMemo(() => new THREE.Vector3(), []);
  const endTarget = useMemo(() => new THREE.Vector3(), []);
  const animRef = useRef({ active: false, t: 0, duration: 0.65 });

  const startTransition = useCallback(
    (mode) => {
      if (!modelBounds || followCamera || immersiveMode) return;

      const { center, size, radius } = modelBounds;
      const horizontal = Math.max(size.x, size.z);
      const distance = Math.max(horizontal * 1.4, radius * 1.85, 8);

      startPos.copy(camera.position);
      if (controlsRef.current) {
        startTarget.copy(controlsRef.current.target);
      } else {
        startTarget.copy(center);
      }
      endTarget.copy(center);

      if (mode === "aerial") {
        const span = Math.max(horizontal, radius * 2);
        endPos.set(center.x + 0.001, center.y + Math.max(size.y + 6, span * 1.55), center.z + 0.001);
        animRef.current.duration = 0.78;
      } else {
        endPos.set(
          center.x + distance * 0.88,
          center.y + Math.max(size.y * 1.18, 5.2),
          center.z + distance * 0.88
        );
        animRef.current.duration = 0.64;
      }

      animRef.current.t = 0;
      animRef.current.active = true;
    },
    [camera, controlsRef, endPos, endTarget, followCamera, immersiveMode, modelBounds, startPos, startTarget]
  );

  useEffect(() => {
    if (!modelBounds || followCamera || immersiveMode) return;
    startTransition("fit");
  }, [followCamera, immersiveMode, modelBounds, startTransition]);

  useEffect(() => {
    if (!presetRequest?.nonce || immersiveMode) return;
    startTransition(presetRequest.mode === "aerial" ? "aerial" : "fit");
  }, [immersiveMode, presetRequest?.mode, presetRequest?.nonce, startTransition]);

  useFrame((_, delta) => {
    if (followCamera || immersiveMode) {
      animRef.current.active = false;
      return;
    }

    if (!animRef.current.active) return;
    animRef.current.t = Math.min(1, animRef.current.t + delta / Math.max(0.001, animRef.current.duration));

    const t = animRef.current.t;
    const ease = 1 - Math.pow(1 - t, 3);
    camera.position.lerpVectors(startPos, endPos, ease);

    if (controlsRef.current) {
      controlsRef.current.target.lerpVectors(startTarget, endTarget, ease);
      controlsRef.current.update();
    } else {
      camera.lookAt(endTarget);
    }

    if (t >= 1) {
      animRef.current.active = false;
    }
  });

  return null;
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
    <mesh position={[pos.x, 0.16, pos.z]} userData={{ noCameraCollision: true }}>
      <sphereGeometry args={[0.14, 20, 20]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.3} />
    </mesh>
  );
}

/**
 * Render movable autonomous agent (non-immersive mode compatibility).
 *
 * @param {object} props
 * @param {{x:number,z:number}|null} props.position
 * @param {number} props.yaw
 */
function AgentMesh({ position, yaw }) {
  const ref = useRef(null);
  const bodyRef = useRef(null);
  const targetPos = useMemo(() => new THREE.Vector3(), []);
  const targetYawRef = useRef(0);
  const currentYawRef = useRef(0);

  useEffect(() => {
    if (!position || !ref.current) return;
    targetPos.set(position.x, 0.2, position.z);
    targetYawRef.current = yaw || 0;
  }, [position, yaw, targetPos]);

  useFrame((_, delta) => {
    if (!ref.current || !position) return;

    const posAlpha = 1 - Math.exp(-delta * 13);
    const yawAlpha = 1 - Math.exp(-delta * 15);

    ref.current.position.lerp(targetPos, posAlpha);

    const currentYaw = currentYawRef.current;
    const targetYaw = targetYawRef.current;
    const deltaYaw = ((targetYaw - currentYaw + Math.PI) % (Math.PI * 2)) - Math.PI;
    currentYawRef.current = currentYaw + deltaYaw * yawAlpha;
    ref.current.rotation.y = currentYawRef.current;

    if (bodyRef.current) {
      const bob = Math.sin(performance.now() * 0.01) * 0.02;
      bodyRef.current.position.y = 0.16 + bob;
    }
  });

  if (!position) return null;

  return (
    <group ref={ref} name="NavispaceAgent" userData={{ noCameraCollision: true }}>
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, 0.02, 0]}
        userData={{ noCameraCollision: true }}
      >
        <ringGeometry args={[0.2, 0.35, 40]} />
        <meshBasicMaterial color="#7cc5ff" transparent opacity={0.2} depthWrite={false} />
      </mesh>

      <mesh ref={bodyRef} castShadow position={[0, 0.16, 0]} userData={{ noCameraCollision: true }}>
        <boxGeometry args={[0.28, 0.22, 0.28]} />
        <meshStandardMaterial color="#f97316" emissive="#7c2d12" emissiveIntensity={0.2} />
      </mesh>
    </group>
  );
}

/**
 * Decorative interior props for immersive atmosphere.
 *
 * @param {object} props
 * @param {number[][]|null} props.occupancyGrid
 * @param {number} props.cellSizeM
 */
function AmbientProps({ occupancyGrid, cellSizeM }) {
  const props = useMemo(() => {
    if (!Array.isArray(occupancyGrid) || !occupancyGrid.length || !cellSizeM) return [];

    let seed = 137;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296;
    };

    const rows = occupancyGrid.length;
    const cols = occupancyGrid[0]?.length || 0;
    const out = [];

    for (let r = 2; r < rows - 2; r += 5) {
      for (let c = 2; c < cols - 2; c += 5) {
        if (occupancyGrid[r]?.[c] !== 0) continue;
        if (rand() < 0.73) continue;

        const jitterX = (rand() - 0.5) * cellSizeM * 2;
        const jitterZ = (rand() - 0.5) * cellSizeM * 2;
        const x = c * cellSizeM + jitterX;
        const z = r * cellSizeM + jitterZ;
        const typeRoll = rand();

        let type = "crate";
        if (typeRoll < 0.35) type = "table";
        else if (typeRoll < 0.62) type = "plant";
        else if (typeRoll < 0.82) type = "bench";

        out.push({
          id: `prop-${r}-${c}`,
          type,
          x,
          z,
          yaw: rand() * Math.PI * 2,
          scale: 0.8 + rand() * 0.6,
        });

        if (out.length >= 120) return out;
      }
    }

    return out;
  }, [occupancyGrid, cellSizeM]);

  if (!props.length) return null;

  return (
    <group>
      {props.map((item) => {
        if (item.type === "table") {
          return (
            <group key={item.id} position={[item.x, 0, item.z]} rotation={[0, item.yaw, 0]} scale={item.scale}>
              <mesh castShadow receiveShadow position={[0, 0.42, 0]} userData={{ noCameraCollision: true }}>
                <boxGeometry args={[0.9, 0.07, 0.6]} />
                <meshStandardMaterial color="#73543c" roughness={0.78} metalness={0.08} />
              </mesh>
              {[-0.36, 0.36].map((lx) =>
                [-0.24, 0.24].map((lz) => (
                  <mesh key={`${item.id}-${lx}-${lz}`} castShadow receiveShadow position={[lx, 0.2, lz]} userData={{ noCameraCollision: true }}>
                    <boxGeometry args={[0.06, 0.4, 0.06]} />
                    <meshStandardMaterial color="#604631" roughness={0.85} metalness={0.04} />
                  </mesh>
                ))
              )}
            </group>
          );
        }

        if (item.type === "plant") {
          return (
            <group key={item.id} position={[item.x, 0, item.z]} rotation={[0, item.yaw, 0]} scale={item.scale}>
              <mesh castShadow receiveShadow position={[0, 0.16, 0]} userData={{ noCameraCollision: true }}>
                <cylinderGeometry args={[0.12, 0.09, 0.22, 12]} />
                <meshStandardMaterial color="#8e6645" roughness={0.9} />
              </mesh>
              <mesh castShadow receiveShadow position={[0, 0.34, 0]} userData={{ noCameraCollision: true }}>
                <sphereGeometry args={[0.18, 14, 14]} />
                <meshStandardMaterial color="#4f7f52" roughness={0.68} metalness={0.03} />
              </mesh>
            </group>
          );
        }

        if (item.type === "bench") {
          return (
            <group key={item.id} position={[item.x, 0, item.z]} rotation={[0, item.yaw, 0]} scale={item.scale}>
              <mesh castShadow receiveShadow position={[0, 0.3, 0]} userData={{ noCameraCollision: true }}>
                <boxGeometry args={[1.0, 0.08, 0.25]} />
                <meshStandardMaterial color="#6b5139" roughness={0.82} />
              </mesh>
              <mesh castShadow receiveShadow position={[0, 0.5, -0.1]} userData={{ noCameraCollision: true }}>
                <boxGeometry args={[1.0, 0.18, 0.07]} />
                <meshStandardMaterial color="#6b5139" roughness={0.82} />
              </mesh>
            </group>
          );
        }

        return (
          <mesh
            key={item.id}
            castShadow
            receiveShadow
            position={[item.x, 0.2, item.z]}
            rotation={[0, item.yaw, 0]}
            scale={item.scale}
            userData={{ noCameraCollision: true }}
          >
            <boxGeometry args={[0.34, 0.34, 0.34]} />
            <meshStandardMaterial color="#7e664d" roughness={0.8} metalness={0.05} />
          </mesh>
        );
      })}
    </group>
  );
}

/**
 * Normalize angle delta to [-PI, PI].
 *
 * @param {number} angle
 * @returns {number}
 */
function normalizeAngle(angle) {
  const twoPi = Math.PI * 2;
  let out = angle;
  while (out > Math.PI) out -= twoPi;
  while (out < -Math.PI) out += twoPi;
  return out;
}

/**
 * Check body collision with radius on occupancy grid.
 *
 * @param {{x:number,z:number}} pos
 * @param {number} radiusM
 * @param {number[][]|null} occupancyGrid
 * @param {number} cellSizeM
 * @returns {boolean}
 */
function isBlockedWithRadius(pos, radiusM, occupancyGrid, cellSizeM) {
  if (!Array.isArray(occupancyGrid) || !occupancyGrid.length || !cellSizeM) return false;

  const rows = occupancyGrid.length;
  const cols = occupancyGrid[0]?.length || 0;
  const maxX = (cols - 1) * cellSizeM;
  const maxZ = (rows - 1) * cellSizeM;

  if (pos.x < 0 || pos.z < 0 || pos.x > maxX || pos.z > maxZ) return true;
  if (isWorldPositionBlocked(pos, occupancyGrid, cellSizeM)) return true;

  const steps = 8;
  for (let i = 0; i < steps; i += 1) {
    const a = (i / steps) * Math.PI * 2;
    const p = {
      x: pos.x + Math.cos(a) * radiusM,
      z: pos.z + Math.sin(a) * radiusM,
    };
    if (isWorldPositionBlocked(p, occupancyGrid, cellSizeM)) return true;
  }

  return false;
}

/**
 * Stylized third-person avatar with idle/walk/run animation states.
 *
 * @param {object} props
 * @param {"idle"|"walk"|"run"} props.animationState
 */
function CharacterAvatar({ animationState }) {
  const bodyRef = useRef(null);
  const leftArmRef = useRef(null);
  const rightArmRef = useRef(null);
  const leftLegRef = useRef(null);
  const rightLegRef = useRef(null);
  const headRef = useRef(null);

  useFrame(({ clock }) => {
    const t = clock.elapsedTime;

    const amp = animationState === "run" ? 0.9 : animationState === "walk" ? 0.55 : 0.08;
    const freq = animationState === "run" ? 11.5 : animationState === "walk" ? 7.2 : 2.0;
    const bobAmp = animationState === "run" ? 0.07 : animationState === "walk" ? 0.04 : 0.016;

    const swing = Math.sin(t * freq) * amp;
    const bob = Math.abs(Math.sin(t * freq * 0.5)) * bobAmp;

    if (leftArmRef.current) leftArmRef.current.rotation.x = swing;
    if (rightArmRef.current) rightArmRef.current.rotation.x = -swing;
    if (leftLegRef.current) leftLegRef.current.rotation.x = -swing;
    if (rightLegRef.current) rightLegRef.current.rotation.x = swing;

    if (bodyRef.current) bodyRef.current.position.y = 0.9 + bob;
    if (headRef.current) {
      headRef.current.position.y = 1.38 + bob * 0.5;
      headRef.current.rotation.y = Math.sin(t * 1.6) * 0.06;
    }
  });

  return (
    <group name="ThirdPersonAvatar" userData={{ noCameraCollision: true }}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]} userData={{ noCameraCollision: true }}>
        <ringGeometry args={[0.24, 0.45, 40]} />
        <meshBasicMaterial color="#6ec2ff" transparent opacity={0.2} depthWrite={false} />
      </mesh>

      <mesh ref={bodyRef} castShadow position={[0, 0.9, 0]} userData={{ noCameraCollision: true }}>
        <capsuleGeometry args={[0.2, 0.46, 8, 14]} />
        <meshStandardMaterial color="#5f8ec7" roughness={0.55} metalness={0.12} />
      </mesh>

      <mesh ref={headRef} castShadow position={[0, 1.38, 0]} userData={{ noCameraCollision: true }}>
        <sphereGeometry args={[0.14, 16, 16]} />
        <meshStandardMaterial color="#d9c8b1" roughness={0.62} />
      </mesh>

      <mesh ref={leftArmRef} castShadow position={[-0.24, 0.96, 0]} userData={{ noCameraCollision: true }}>
        <boxGeometry args={[0.11, 0.4, 0.11]} />
        <meshStandardMaterial color="#3d5f89" roughness={0.65} />
      </mesh>
      <mesh ref={rightArmRef} castShadow position={[0.24, 0.96, 0]} userData={{ noCameraCollision: true }}>
        <boxGeometry args={[0.11, 0.4, 0.11]} />
        <meshStandardMaterial color="#3d5f89" roughness={0.65} />
      </mesh>

      <mesh ref={leftLegRef} castShadow position={[-0.1, 0.36, 0]} userData={{ noCameraCollision: true }}>
        <boxGeometry args={[0.12, 0.46, 0.12]} />
        <meshStandardMaterial color="#2c3f58" roughness={0.75} />
      </mesh>
      <mesh ref={rightLegRef} castShadow position={[0.1, 0.36, 0]} userData={{ noCameraCollision: true }}>
        <boxGeometry args={[0.12, 0.46, 0.12]} />
        <meshStandardMaterial color="#2c3f58" roughness={0.75} />
      </mesh>
    </group>
  );
}

/**
 * Third-person WASD movement + collision controller.
 *
 * @param {object} props
 * @param {boolean} props.enabled
 * @param {{x:number,y?:number,z:number}|null} props.initialPosition
 * @param {number[][]|null} props.occupancyGrid
 * @param {number} props.cellSizeM
 * @param {(data:{position:{x:number,y:number,z:number},yaw:number,speed:number,animation:"idle"|"walk"|"run"})=>void} [props.onTelemetry]
 */
function ThirdPersonController({
  enabled,
  initialPosition,
  occupancyGrid,
  cellSizeM,
  onTelemetry,
}) {
  const groupRef = useRef(null);
  const keysRef = useRef({ w: false, a: false, s: false, d: false, shift: false });
  const velocityRef = useRef(new THREE.Vector3());
  const yawRef = useRef(0);
  const animRef = useRef("idle");
  const lastEmitRef = useRef(0);

  const [animationState, setAnimationState] = useState("idle");

  const forward = useMemo(() => new THREE.Vector3(), []);
  const right = useMemo(() => new THREE.Vector3(), []);
  const move = useMemo(() => new THREE.Vector3(), []);
  const targetVelocity = useMemo(() => new THREE.Vector3(), []);
  const step = useMemo(() => new THREE.Vector3(), []);

  useEffect(() => {
    const onKeyDown = (event) => {
      const tag = String(event.target?.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;

      if (event.code === "KeyW") keysRef.current.w = true;
      if (event.code === "KeyA") keysRef.current.a = true;
      if (event.code === "KeyS") keysRef.current.s = true;
      if (event.code === "KeyD") keysRef.current.d = true;
      if (event.code === "ShiftLeft" || event.code === "ShiftRight") keysRef.current.shift = true;
    };

    const onKeyUp = (event) => {
      if (event.code === "KeyW") keysRef.current.w = false;
      if (event.code === "KeyA") keysRef.current.a = false;
      if (event.code === "KeyS") keysRef.current.s = false;
      if (event.code === "KeyD") keysRef.current.d = false;
      if (event.code === "ShiftLeft" || event.code === "ShiftRight") keysRef.current.shift = false;
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, []);

  useEffect(() => {
    if (!groupRef.current || !initialPosition) return;

    groupRef.current.position.set(initialPosition.x, 0, initialPosition.z);
    yawRef.current = 0;
    groupRef.current.rotation.y = yawRef.current;
    velocityRef.current.set(0, 0, 0);

    onTelemetry?.({
      position: { x: initialPosition.x, y: 0.2, z: initialPosition.z },
      yaw: yawRef.current,
      speed: 0,
      animation: "idle",
    });
  }, [initialPosition, onTelemetry]);

  useFrame(({ camera, clock }, delta) => {
    if (!enabled || !groupRef.current) return;

    // Movement basis is camera-facing for intuitive third-person control.
    camera.getWorldDirection(forward);
    forward.y = 0;
    if (forward.lengthSq() < 1e-6) {
      forward.set(1, 0, 0);
    }
    forward.normalize();

    right.set(forward.z, 0, -forward.x).normalize();

    move.set(0, 0, 0);
    if (keysRef.current.w) move.add(forward);
    if (keysRef.current.s) move.sub(forward);
    if (keysRef.current.d) move.add(right);
    if (keysRef.current.a) move.sub(right);

    const hasInput = move.lengthSq() > 1e-6;
    const isRunning = keysRef.current.shift && hasInput;
    const targetSpeed = hasInput ? (isRunning ? 3.4 : 1.9) : 0;

    if (hasInput) move.normalize();
    targetVelocity.copy(move).multiplyScalar(targetSpeed);

    const velLerp = 1 - Math.exp(-delta * 10.5);
    velocityRef.current.lerp(targetVelocity, velLerp);

    // Extra damping when no movement keys are active.
    if (!hasInput) {
      velocityRef.current.multiplyScalar(Math.exp(-delta * 5.5));
    }

    const currentPos = groupRef.current.position;
    step.copy(velocityRef.current).multiplyScalar(delta);

    // Axis-separated collision with simple radius body test.
    const radiusM = 0.24;

    const candidateX = { x: currentPos.x + step.x, z: currentPos.z };
    if (!isBlockedWithRadius(candidateX, radiusM, occupancyGrid, cellSizeM)) {
      currentPos.x = candidateX.x;
    } else {
      velocityRef.current.x = 0;
    }

    const candidateZ = { x: currentPos.x, z: currentPos.z + step.z };
    if (!isBlockedWithRadius(candidateZ, radiusM, occupancyGrid, cellSizeM)) {
      currentPos.z = candidateZ.z;
    } else {
      velocityRef.current.z = 0;
    }

    const horizontalSpeed = Math.hypot(velocityRef.current.x, velocityRef.current.z);

    if (horizontalSpeed > 0.05) {
      const targetYaw = Math.atan2(velocityRef.current.z, velocityRef.current.x);
      const yawDelta = normalizeAngle(targetYaw - yawRef.current);
      yawRef.current += yawDelta * (1 - Math.exp(-delta * 14));
      groupRef.current.rotation.y = yawRef.current;
    }

    let nextAnim = "idle";
    if (horizontalSpeed > 0.1) {
      nextAnim = isRunning ? "run" : "walk";
    }

    if (nextAnim !== animRef.current) {
      animRef.current = nextAnim;
      setAnimationState(nextAnim);
    }

    const nowMs = clock.elapsedTime * 1000;
    if (nowMs - lastEmitRef.current > 90 || nextAnim !== animRef.current) {
      lastEmitRef.current = nowMs;
      onTelemetry?.({
        position: { x: currentPos.x, y: 0.2, z: currentPos.z },
        yaw: yawRef.current,
        speed: horizontalSpeed,
        animation: nextAnim,
      });
    }
  });

  return (
    <group ref={groupRef} userData={{ noCameraCollision: true }}>
      <CharacterAvatar animationState={animationState} />
    </group>
  );
}

/**
 * Optional camera follow behavior that tracks the subject.
 *
 * @param {object} props
 * @param {boolean} props.enabled
 * @param {{x:number,z:number,y?:number}|null} props.position
 * @param {number} props.yaw
 */
function FollowCamera({ enabled, position, yaw }) {
  const { camera, scene } = useThree();
  const raycaster = useMemo(() => new THREE.Raycaster(), []);
  const lookTarget = useMemo(() => new THREE.Vector3(), []);
  const smoothLook = useMemo(() => new THREE.Vector3(), []);
  const dir = useMemo(() => new THREE.Vector3(), []);
  const candidates = useMemo(
    () => [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()],
    []
  );
  const adjusted = useMemo(
    () => [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()],
    []
  );
  const didInit = useRef(false);

  /**
   * Keep line-of-sight by moving the camera in front of blocking geometry.
   *
   * @param {THREE.Vector3} candidate
   * @param {THREE.Vector3} out
   */
  function adjustForOcclusion(candidate, out) {
    dir.copy(candidate).sub(lookTarget);
    const maxDistance = dir.length();

    if (maxDistance < 1e-4) {
      out.copy(candidate);
      return;
    }

    dir.normalize();
    raycaster.set(lookTarget, dir);
    raycaster.near = 0.12;
    raycaster.far = maxDistance;

    const hits = raycaster.intersectObjects(scene.children, true).filter((hit) => {
      const obj = hit.object;
      return obj?.isMesh && obj.visible && !obj.userData?.noCameraCollision && hit.distance > 0.16;
    });

    if (!hits.length) {
      out.copy(candidate);
      return;
    }

    const safeDistance = Math.max(0.55, hits[0].distance - 0.28);
    out.copy(lookTarget).addScaledVector(dir, safeDistance);
  }

  useFrame((_, delta) => {
    if (!enabled || !position) {
      didInit.current = false;
      return;
    }

    const headingX = Math.cos(yaw || 0);
    const headingZ = Math.sin(yaw || 0);
    const sideX = -headingZ;
    const sideZ = headingX;

    lookTarget.set(position.x, (position.y ?? 0.2) + 0.95, position.z);

    candidates[0].set(
      lookTarget.x - headingX * 3.1 + sideX * 0.75,
      lookTarget.y + 1.65,
      lookTarget.z - headingZ * 3.1 + sideZ * 0.75
    );
    candidates[1].set(
      lookTarget.x - headingX * 2.7 + sideX * 1.25,
      lookTarget.y + 2.2,
      lookTarget.z - headingZ * 2.7 + sideZ * 1.25
    );
    candidates[2].set(
      lookTarget.x - headingX * 3.2 - sideX * 1.05,
      lookTarget.y + 1.85,
      lookTarget.z - headingZ * 3.2 - sideZ * 1.05
    );
    candidates[3].set(
      lookTarget.x - headingX * 2.2,
      lookTarget.y + 2.6,
      lookTarget.z - headingZ * 2.2
    );

    let bestIndex = 0;
    let bestScore = -1;

    for (let i = 0; i < candidates.length; i += 1) {
      adjustForOcclusion(candidates[i], adjusted[i]);
      const score = adjusted[i].distanceToSquared(lookTarget);
      if (score > bestScore) {
        bestScore = score;
        bestIndex = i;
      }
    }

    const chosen = adjusted[bestIndex];
    const posLerp = 1 - Math.exp(-delta * 7.2);
    const lookLerp = 1 - Math.exp(-delta * 10.5);

    if (!didInit.current) {
      camera.position.copy(chosen);
      smoothLook.copy(lookTarget);
      didInit.current = true;
    } else {
      camera.position.lerp(chosen, posLerp);
      smoothLook.lerp(lookTarget, lookLerp);
    }

    camera.lookAt(smoothLook);
  });

  return null;
}

/**
 * SceneCanvas renders the 3D navigation environment.
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
 * @param {{mode:"fit"|"aerial",nonce:number}|null} props.cameraPresetRequest
 * @param {(payload:{cell:{row:number,col:number},world:{x:number,z:number}})=>void} props.onCellPick
 * @param {boolean} [props.immersiveMode=false]
 * @param {boolean} [props.thirdPersonEnabled=false]
 * @param {{x:number,y?:number,z:number}|null} [props.initialPlayerPosition=null]
 * @param {(data:{position:{x:number,y:number,z:number},yaw:number,speed:number,animation:string})=>void} [props.onPlayerTelemetry]
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
  cameraPresetRequest,
  onCellPick,
  immersiveMode = false,
  thirdPersonEnabled = false,
  initialPlayerPosition = null,
  onPlayerTelemetry,
}) {
  const [modelBounds, setModelBounds] = useState(null);
  const [controllerTelemetry, setControllerTelemetry] = useState({
    position: null,
    yaw: 0,
    animation: "idle",
    speed: 0,
  });

  const controlsRef = useRef(null);

  const handleModelBounds = useCallback((bounds) => {
    setModelBounds(bounds);
  }, []);

  const handleControllerTelemetry = useCallback(
    (data) => {
      setControllerTelemetry(data);
      onPlayerTelemetry?.(data);
    },
    [onPlayerTelemetry]
  );

  const gridShape = useMemo(() => {
    if (!Array.isArray(occupancyGrid) || !occupancyGrid.length) return null;
    return { rows: occupancyGrid.length, cols: occupancyGrid[0]?.length || 0 };
  }, [occupancyGrid]);

  /**
   * Handle click on floor plane and map to occupancy grid coordinates.
   * @param {import('@react-three/fiber').ThreeEvent<MouseEvent>} event
   */
  function handlePlaneClick(event) {
    if (immersiveMode || !gridShape || !cellSizeM) return;

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

  const orbitLimits = useMemo(() => {
    if (!modelBounds) {
      return {
        minDistance: 2,
        maxDistance: 140,
      };
    }
    return {
      minDistance: Math.max(0.6, Math.min(4, modelBounds.radius * 0.04)),
      maxDistance: Math.max(28, modelBounds.radius * 8.0),
    };
  }, [modelBounds]);

  const activeCameraPosition = immersiveMode ? controllerTelemetry.position : agentPosition;
  const activeCameraYaw = immersiveMode ? controllerTelemetry.yaw : agentYaw;

  return (
    <div className="scene-shell" aria-label="3D navigation scene">
      <Canvas camera={{ position: [11, 9, 11], fov: 58, near: 0.2, far: 420 }} gl={{ antialias: true, logarithmicDepthBuffer: true }}>
        <color attach="background" args={immersiveMode ? ["#a6b4c3"] : ["#c8d8eb"]} />
        <fog attach="fog" args={immersiveMode ? ["#a6b4c3", 16, 180] : ["#c8d8eb", 28, 170]} />

        <ambientLight intensity={immersiveMode ? 0.42 : 0.6} />
        <hemisphereLight intensity={immersiveMode ? 0.55 : 0.45} color="#ffffff" groundColor="#8d9eb2" />
        <directionalLight position={[12, 15, 8]} intensity={immersiveMode ? 1.35 : 1.2} castShadow />
        <directionalLight position={[-8, 10, -6]} intensity={immersiveMode ? 0.48 : 0.35} />

        {immersiveMode ? (
          <spotLight
            position={[planeCenter[0], 8, planeCenter[2]]}
            angle={0.7}
            penumbra={0.38}
            intensity={0.45}
            castShadow
          />
        ) : null}

        <mesh
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, -0.12, 0]}
          receiveShadow
          renderOrder={-30}
          userData={{ noCameraCollision: true }}
        >
          <planeGeometry args={[500, 500]} />
          <meshStandardMaterial color={immersiveMode ? "#8f9dab" : "#dbe7f2"} roughness={0.92} metalness={0.02} />
        </mesh>

        <mesh
          rotation={[-Math.PI / 2, 0, 0]}
          position={[planeCenter[0], -0.06, planeCenter[2]]}
          receiveShadow
          renderOrder={-20}
          userData={{ noCameraCollision: true }}
        >
          <planeGeometry args={planeSize} />
          <meshStandardMaterial
            color={immersiveMode ? "#9aa8b5" : "#f3f8ff"}
            transparent
            opacity={immersiveMode ? 0.34 : 0.26}
            roughness={0.92}
            depthWrite={false}
            polygonOffset
            polygonOffsetFactor={-1}
            polygonOffsetUnits={-1}
          />
        </mesh>

        <Grid
          args={[200, 200]}
          cellSize={1}
          cellThickness={immersiveMode ? 0.24 : 0.42}
          sectionSize={10}
          sectionThickness={immersiveMode ? 0.62 : 0.95}
          fadeDistance={100}
          fadeStrength={1}
          position={[0, -0.055, 0]}
          cellColor={immersiveMode ? "#8ea0b6" : "#7f95ad"}
          sectionColor={immersiveMode ? "#5f7b99" : "#3875c8"}
        />

        <mesh
          rotation={[-Math.PI / 2, 0, 0]}
          position={planeCenter}
          onClick={handlePlaneClick}
          visible={false}
          userData={{ noCameraCollision: true }}
        >
          <planeGeometry args={planeSize} />
          <meshBasicMaterial transparent opacity={0} />
        </mesh>

        {immersiveMode ? <AmbientProps occupancyGrid={occupancyGrid} cellSizeM={cellSizeM} /> : null}

        <Suspense fallback={null}>
          {modelUrl ? <BuildingModel url={modelUrl} onBounds={handleModelBounds} /> : null}
        </Suspense>

        {!immersiveMode ? <CellMarker cell={startCell} cellSizeM={cellSizeM} color="#22c55e" /> : null}
        {!immersiveMode ? <CellMarker cell={goalCell} cellSizeM={cellSizeM} color="#ef4444" /> : null}

        {!immersiveMode ? <PathRenderer path={pathWorld} /> : null}

        {immersiveMode && thirdPersonEnabled ? (
          <ThirdPersonController
            enabled
            initialPosition={initialPlayerPosition}
            occupancyGrid={occupancyGrid}
            cellSizeM={cellSizeM}
            onTelemetry={handleControllerTelemetry}
          />
        ) : (
          <AgentMesh position={agentPosition} yaw={agentYaw} />
        )}

        <FollowCamera enabled={immersiveMode ? true : followCamera} position={activeCameraPosition} yaw={activeCameraYaw} />

        <ViewController
          modelBounds={modelBounds}
          followCamera={followCamera}
          immersiveMode={immersiveMode}
          controlsRef={controlsRef}
          presetRequest={cameraPresetRequest}
        />

        <axesHelper args={[1.5]} />
        <OrbitControls
          ref={controlsRef}
          makeDefault
          enabled={!followCamera && !immersiveMode}
          enableDamping
          dampingFactor={0.09}
          minDistance={orbitLimits.minDistance}
          maxDistance={orbitLimits.maxDistance}
          minPolarAngle={0.08}
          maxPolarAngle={Math.PI / 2 - 0.02}
        />
      </Canvas>
    </div>
  );
}
