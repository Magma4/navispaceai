/**
 * Three.js scene setup and rendering helpers.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

export class NavScene {
  /**
   * @param {HTMLElement} container - DOM container for the WebGL canvas.
   */
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#d9e4ec");

    this.camera = new THREE.PerspectiveCamera(
      60,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    this.camera.position.set(12, 10, 12);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(5, 0, 5);
    this.controls.enableDamping = true;

    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    this.modelRoot = null;
    this.pathLine = null;
    this.agent = null;
    this.clickEnabled = false;
    this.clickCallback = null;
    this.followAgent = false;

    this._buildEnvironment();
    this._bindEvents();
    this._animate();
  }

  /** Build static scene elements: lights, floor helper, and pick plane. */
  _buildEnvironment() {
    const hemi = new THREE.HemisphereLight(0xffffff, 0x445566, 0.85);
    this.scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(10, 20, 8);
    this.scene.add(key);

    const grid = new THREE.GridHelper(300, 300, 0x7c8fa3, 0x9ab0c4);
    this.scene.add(grid);

    // Invisible floor plane used for click-to-point raycasting.
    const planeGeo = new THREE.PlaneGeometry(1000, 1000);
    const planeMat = new THREE.MeshBasicMaterial({ visible: false });
    this.pickPlane = new THREE.Mesh(planeGeo, planeMat);
    this.pickPlane.rotateX(-Math.PI / 2);
    this.scene.add(this.pickPlane);
  }

  /** Register resize and pointer handlers. */
  _bindEvents() {
    window.addEventListener("resize", () => {
      this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    });

    this.renderer.domElement.addEventListener("pointerdown", (event) => {
      if (!this.clickEnabled || !this.clickCallback) return;

      const rect = this.renderer.domElement.getBoundingClientRect();
      this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      this.raycaster.setFromCamera(this.mouse, this.camera);
      const hits = this.raycaster.intersectObject(this.pickPlane, false);
      if (hits.length === 0) return;

      const p = hits[0].point;
      this.clickCallback({ x: p.x, y: p.y, z: p.z });
    });
  }

  /** Main render loop. */
  _animate() {
    requestAnimationFrame(() => this._animate());
    this.controls.update();

    if (this.followAgent && this.agent) {
      const target = this.agent.position.clone();
      const desiredCam = target.clone().add(new THREE.Vector3(2.8, 2.2, 2.8));
      this.camera.position.lerp(desiredCam, 0.07);
      this.controls.target.lerp(target, 0.08);
    }

    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Load a glTF/glb model into the scene.
   * @param {string} modelUrl
   * @returns {Promise<void>}
   */
  async loadModel(modelUrl) {
    const loader = new GLTFLoader();
    if (this.modelRoot) {
      this.scene.remove(this.modelRoot);
      this.modelRoot.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
          else obj.material.dispose();
        }
      });
      this.modelRoot = null;
    }

    const gltf = await loader.loadAsync(modelUrl);
    this.modelRoot = gltf.scene;

    // Enforce semantic coloring by mesh name so walls/doors/stairs are visually distinct.
    this.modelRoot.traverse((obj) => {
      if (!obj.isMesh || !obj.material) return;
      const name = String(obj.name || "").toLowerCase();

      let color = null;
      if (name.startsWith("door_")) color = "#e67828";
      else if (name.startsWith("stair_")) color = "#4eaaff";
      else if (name.startsWith("wall_")) color = "#5c6d82";
      else if (name === "floor") color = "#c0cbd6";

      if (color) {
        const base = new THREE.MeshStandardMaterial({
          color,
          roughness: 0.55,
          metalness: 0.05,
        });
        if (name.startsWith("door_") || name.startsWith("stair_")) {
          base.emissive = new THREE.Color(color);
          base.emissiveIntensity = 0.12;
        }
        if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
        else obj.material.dispose();
        obj.material = base;
      }
    });

    this.scene.add(this.modelRoot);
  }

  /**
   * Draw a world-space path line.
   * @param {Array<{x:number,z:number}>} worldPath
   */
  setPath(worldPath) {
    if (this.pathLine) {
      this.scene.remove(this.pathLine);
      this.pathLine.geometry.dispose();
      this.pathLine.material.dispose();
      this.pathLine = null;
    }
    if (!worldPath || worldPath.length < 2) return;

    const points = worldPath.map((p) => new THREE.Vector3(p.x, 0.08, p.z));
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color: 0x2e7d32 });
    this.pathLine = new THREE.Line(geometry, material);
    this.scene.add(this.pathLine);
  }

  /**
   * Create or reset the autonomous agent mesh.
   */
  ensureAgent() {
    if (!this.agent) {
      const geo = new THREE.SphereGeometry(0.18, 24, 24);
      const mat = new THREE.MeshStandardMaterial({ color: 0xd84315, roughness: 0.4 });
      this.agent = new THREE.Mesh(geo, mat);
      this.agent.position.set(0, 0.18, 0);
      this.scene.add(this.agent);
    }
  }

  /**
   * Move agent to position and orient it by yaw.
   * @param {{x:number,z:number}} pos
   * @param {number} yaw
   */
  updateAgent(pos, yaw) {
    this.ensureAgent();
    this.agent.position.set(pos.x, 0.18, pos.z);
    this.agent.rotation.y = yaw;
  }

  /**
   * Enable or disable scene picking mode.
   * @param {boolean} enabled
   * @param {(point:{x:number,y:number,z:number})=>void} callback
   */
  setClickMode(enabled, callback = null) {
    this.clickEnabled = enabled;
    this.clickCallback = callback;
  }

  /**
   * Toggle camera-follow mode for agent.
   * @param {boolean} enabled
   */
  setFollowAgent(enabled) {
    this.followAgent = enabled;
  }
}
