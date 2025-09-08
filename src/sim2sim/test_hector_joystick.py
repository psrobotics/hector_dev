import argparse
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple, Deque
from collections import deque
import numpy as np
import onnxruntime as rt
import mujoco
import imageio
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt

from etils import epath
from mujoco_playground._src import mjx_env
from hector_pg import constants as hector_constants

# ------------ Controller (adapted from your script) ------------
class OnnxController:
    """ONNX controller with action-delay, obs history, and phase clock."""

    def __init__(
        self,
        policy_path: str,
        default_angles: np.ndarray,
        ctrl_dt: float,
        n_substeps: int,
        action_scale: float = 0.6,
        obs_size: int = 67,
        obs_hist: int = 5,
        gait_freq: float = 1.8,
        action_delay_ticks: int = 0,   # latency measured in control ticks
        init_wait_steps: int = 80,
        constant_command: Sequence[float] = (0.0, 0.0, 0.0),
    ):
        self._output_names = ["continuous_actions"]
        self._policy = rt.InferenceSession(
            policy_path, providers=["CPUExecutionProvider"]
        )
        self._action_scale = float(action_scale)
        self._default_angles = default_angles.astype(np.float32)
        self._last_action = np.zeros_like(default_angles, dtype=np.float32)

        self._obs_size = int(obs_size)
        self._obs_hist = int(obs_hist)
        self._obs_buffer = np.zeros(self._obs_size * self._obs_hist, dtype=np.float32)

        self._counter = 0
        self._n_substeps = int(n_substeps)

        # Phase clock for two legs (0 and pi)
        self._phase = np.array([0.0, np.pi], dtype=np.float32)
        self._gait_freq = float(gait_freq)
        self._phase_dt = 2.0 * np.pi * self._gait_freq * ctrl_dt

        # Latency buffer for action delay (FIFO of actions to apply later)
        self._action_delay_ticks = max(0, int(action_delay_ticks))
        self._action_fifo: Deque[np.ndarray] = deque(
            [self._default_angles.copy() for _ in range(self._action_delay_ticks)],
            maxlen=self._action_delay_ticks,
        )

        self._init_wait_steps = int(init_wait_steps)
        self._command = np.asarray(constant_command, dtype=np.float32)

        # fixed observation map: gyro, gravity-from-root xmat, joints, vels, last_action, phase, command
        # ensure sizes line up with obs_size you configured

    def _get_obs_once(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        gyro = data.sensor("gyro").data  # shape (3,)
        imu_xmat = data.site_xmat[model.site("root").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1], dtype=np.float32)  # (3,)
        joint_angles = (data.qpos[7:] - self._default_angles).astype(np.float32)
        joint_velocities = data.qvel[6:].astype(np.float32)

        phase = np.concatenate([np.cos(self._phase), np.sin(self._phase)]).astype(np.float32)

        obs_n = np.hstack([
            gyro.astype(np.float32),
            gravity.astype(np.float32),
            joint_angles,
            joint_velocities,
            self._last_action.astype(np.float32),
            phase,
            self._command,
        ]).astype(np.float32)

        if obs_n.size != self._obs_size:
            raise RuntimeError(
                f"obs_n has size {obs_n.size}, but expected obs_size={self._obs_size}. "
                f"Adjust fields or obs_size to match."
            )
        return obs_n

    def get_obs(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        obs_n = self._get_obs_once(model, data)
        # Stack history at the *front* (newest first)
        obs = np.hstack([obs_n, self._obs_buffer]).astype(np.float32)
        # Update history buffer (drop oldest obs_size slice)
        self._obs_buffer = np.hstack([obs_n, self._obs_buffer[:-self._obs_size]]).astype(np.float32)
        return obs

    def step_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Run every control tick (each ctrl_dt)."""
        # Build observation (with history)
        obs = self.get_obs(model, data)
        onnx_input = {"obs": obs.reshape(1, -1)}
        onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0].astype(np.float32)
        self._last_action = onnx_pred.copy()

        # Scale to joint target
        target_ctrl = onnx_pred * self._action_scale + self._default_angles

        # Apply action delay if requested
        if self._action_delay_ticks > 0:
            # push new action and pop the one to apply
            self._action_fifo.append(target_ctrl)
            apply_ctrl = self._action_fifo.popleft()
        else:
            apply_ctrl = target_ctrl

        data.ctrl[:] = apply_ctrl

        # Stabilize initial pose for a while
        if self._counter < self._init_wait_steps:
            data.qpos[0:2] = 0.0
            data.qpos[2] = 0.55
            data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            data.qvel[0:6] = 0.0
            data.ctrl[:] = self._default_angles

        # Advance phase
        next_phase = self._phase + self._phase_dt
        self._phase = (np.fmod(next_phase + np.pi, 2.0 * np.pi) - np.pi).astype(np.float32)

        self._counter += 1


# ------------ Batch runner ------------
@dataclass
class BatchConfig:
    onnx_paths: List[Path]
    out_dir: Path
    seconds: float = 20.0
    ctrl_dt: float = 0.02
    sim_dt: float = 0.002
    action_scale: float = 0.60
    obs_size: int = 67
    obs_hist: int = 20
    gait_freq: float = 1.8
    init_wait_steps: int = 80

    # latency sweep (integer control ticks)
    delays: List[int] = None

    # disturbance config
    disturb_time: float = 5.0       # seconds from start
    disturb_duration: float = 0.1   # seconds
    disturb_dv: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # delta on base linear velocity (x,y,z)

    # render
    width: int = 1920
    height: int = 1080
    video_fps: int = 50


def load_model_and_data() -> Tuple[mujoco.MjModel, mujoco.MjData]:
    assets = {}
    mjx_env.update_assets(assets, hector_constants.ROOT_PATH / "xmls", "*.xml")
    mjx_env.update_assets(assets, hector_constants.ROOT_PATH / "xmls" / "meshes")
    model = mujoco.MjModel.from_xml_path(
        hector_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(), assets
    )
    data = mujoco.MjData(model)

    # reset to keyframe 'home' index 1 (like your code)
    mujoco.mj_resetDataKeyframe(model, data, 1)
    return model, data


def run_single(
    onnx_path: Path,
    delay_ticks: int,
    cfg: BatchConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a single simulation with a given policy and delay. Returns t, base_xyz arrays."""
    model, data = load_model_and_data()
    sim_dt = cfg.sim_dt
    ctrl_dt = cfg.ctrl_dt
    n_substeps = int(round(ctrl_dt / sim_dt))
    model.opt.timestep = sim_dt

    # default angles from keyframe 'home'
    default_angles = np.array(model.keyframe("home").qpos[7:], dtype=np.float32)

    controller = OnnxController(
        policy_path=str(onnx_path),
        default_angles=default_angles,
        ctrl_dt=ctrl_dt,
        n_substeps=n_substeps,
        action_scale=cfg.action_scale,
        obs_size=cfg.obs_size,
        obs_hist=cfg.obs_hist,
        gait_freq=cfg.gait_freq,
        action_delay_ticks=delay_ticks,
        init_wait_steps=cfg.init_wait_steps,
        constant_command=(0.0, 0.0, 0.0),
    )

    # First set control to default
    data.ctrl[:] = controller._default_angles

    # Timing
    total_steps = int(round(cfg.seconds / sim_dt))
    ctrl_every = n_substeps  # call controller every n_substeps
    disturb_start_step = int(round(cfg.disturb_time / sim_dt))
    disturb_end_step = disturb_start_step + int(round(cfg.disturb_duration / sim_dt))
    dv = np.asarray(cfg.disturb_dv, dtype=np.float32)

    # Offline renderer
    renderer = mujoco.Renderer(model, height=cfg.height, width=cfg.width)
    # Camera: lock to a named camera if you have it, else default free cam is OK

    # Video writer
    out_base = f"{onnx_path.stem}_delay{delay_ticks}t"
    mp4_path = cfg.out_dir / f"{out_base}.mp4"
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(mp4_path.as_posix(), fps=cfg.video_fps, codec="libx264", quality=8)

    # Logs
    base_xyz = np.zeros((total_steps, 3), dtype=np.float32)
    time_s = np.arange(total_steps, dtype=np.float32) * sim_dt

    # Main loop
    for step in range(total_steps):
        # Disturbance: modify base linear velocity directly
        if disturb_start_step <= step < disturb_end_step:
            data.qvel[0:3] += dv  # apply once per step inside window (intentional "step-like" kick)

        # Control tick?
        if step % ctrl_every == 0:
            controller.step_control(model, data)
            # Render a frame to video (subsample to video fps)
            # Keep it simple: render every (sim_dt) but writer will just get all frames.
            renderer.update_scene(data)
            img = renderer.render()
            writer.append_data(img)
            
        # Step sim
        mujoco.mj_step(model, data)

        # Record base position
        base_xyz[step] = data.qpos[0:3]

    writer.close()
    renderer.close()

    return time_s, base_xyz[:, 0], base_xyz[:, 1], base_xyz[:, 2]


def plot_base(time_s: np.ndarray,
              x: np.ndarray, y: np.ndarray, z: np.ndarray,
              disturb_time: float,
              out_png: Path,
              title: str):
    plt.figure(figsize=(8, 5))
    plt.subplot(3, 1, 1)
    plt.plot(time_s, x)
    plt.axvline(disturb_time, linestyle="--")
    plt.ylabel("x (m)")
    plt.title(title)

    plt.subplot(3, 1, 2)
    plt.plot(time_s, y)
    plt.axvline(disturb_time, linestyle="--")
    plt.ylabel("y (m)")

    plt.subplot(3, 1, 3)
    plt.plot(time_s, z)
    plt.axvline(disturb_time, linestyle="--")
    plt.ylabel("z (m)")
    plt.xlabel("time (s)")

    plt.tight_layout()
    plt.savefig(out_png.as_posix(), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", type=str, default=str((epath.Path(__file__).parent / "onnx" /"test").as_posix()),
                        help="Directory containing *.onnx files to test")
    parser.add_argument("--out_dir", type=str, default="batch_out", help="Output directory for videos/plots")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--sim_dt", type=float, default=0.002)
    parser.add_argument("--action_scale", type=float, default=0.60)
    parser.add_argument("--obs_size", type=int, default=67)
    parser.add_argument("--obs_hist", type=int, default=40)
    parser.add_argument("--gait_freq", type=float, default=1.8)
    parser.add_argument("--init_wait_steps", type=int, default=40)
    parser.add_argument("--delays", type=str, default="0,2,4,6",
                        help="Comma-separated integers for action-delay ticks")
    parser.add_argument("--disturb_time", type=float, default=6.0)
    parser.add_argument("--disturb_duration", type=float, default=0.01)
    parser.add_argument("--disturb_dv", type=str, default="0.2,0.0,0.0",
                        help="Comma-separated delta linear velocity (m/s) applied to qvel[0:3]")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--video_fps", type=int, default=50)

    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    out_dir = Path(args.out_dir)
    onnx_paths = sorted(onnx_dir.glob("*.onnx"))
    if not onnx_paths:
        raise FileNotFoundError(f"No ONNX files found in {onnx_dir}")

    delays = [int(x.strip()) for x in args.delays.split(",") if x.strip() != ""]
    dv = tuple(float(x.strip()) for x in args.disturb_dv.split(","))

    cfg = BatchConfig(
        onnx_paths=onnx_paths,
        out_dir=out_dir,
        seconds=args.seconds,
        ctrl_dt=args.ctrl_dt,
        sim_dt=args.sim_dt,
        action_scale=args.action_scale,
        obs_size=args.obs_size,
        obs_hist=args.obs_hist,
        gait_freq=args.gait_freq,
        init_wait_steps=args.init_wait_steps,
        delays=delays,
        disturb_time=args.disturb_time,
        disturb_duration=args.disturb_duration,
        disturb_dv=dv,
        width=args.width,
        height=args.height,
        video_fps=args.video_fps,
    )

    # Sweep policies × delays
    for onnx_path in cfg.onnx_paths:
        for d in cfg.delays:
            print(f"[RUN] {onnx_path.name} | delay={d} ticks")
            t, x, y, z = run_single(onnx_path, d, cfg)

            # Plot base position
            out_base = f"{onnx_path.stem}_delay{d}t"
            png_path = cfg.out_dir / f"{out_base}_base.png"
            png_path.parent.mkdir(parents=True, exist_ok=True)

            title = f"{onnx_path.stem} | delay={d} ticks | disturb @ {cfg.disturb_time:.2f}s"
            plot_base(t, x, y, z, cfg.disturb_time, png_path, title)

    print(f"Done. Outputs in: {cfg.out_dir.resolve()}")


if __name__ == "__main__":
    main()
