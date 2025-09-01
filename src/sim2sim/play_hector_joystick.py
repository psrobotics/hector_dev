"""Deploy an MJX policy in ONNX format to C MuJoCo and play with it."""

from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt

from hector_pg import constants as hector_constants
from mujoco_playground._src import mjx_env
#from mujoco_playground.experimental.sim2sim.gamepad_reader import Gamepad

# lcm external setup
import lcm
from lcm_t.exlcm import twist_t
import os
import select

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / 'onnx' / 'joystick'

vel_cmd_global = np.array([0.0, 0.0, 0.0], dtype=np.float32)
def lcm_handle(channel, data):
    msg = twist_t.decode(data)
    
    def map_deadzone(_in, min=1.0, max=2.0):
      if abs(_in) < 0.1:
        return 0.0
      dir = np.sign(_in)
      out = dir * (abs(_in) + min)
      return np.clip(out, -max, max)

    vel_cmd_global[0] = map_deadzone(msg.x_vel[0], min=0.1, max=1.2)
    vel_cmd_global[1] = map_deadzone(msg.y_vel[0], min=0.1, max=1.2)
    vel_cmd_global[2] = map_deadzone(msg.omega_vel[0], min=0.1, max=1.2)
    
    print(vel_cmd_global)



class OnnxController:
  """ONNX controller for the Hector."""

  def __init__(
      self,
      policy_path: str,
      default_angles: np.ndarray,
      ctrl_dt: float,
      n_substeps: int,
      action_scale: float = 0.5,
  ):
    self._output_names = ["continuous_actions"]
    self._policy = rt.InferenceSession(
        policy_path, providers=["CPUExecutionProvider"]
    )

    self._action_scale = action_scale
    self._default_angles = default_angles
    self._last_action = np.zeros_like(default_angles, dtype=np.float32)
    
    self._obs_size = 67
    self._last_obs_1 = np.zeros(self._obs_size, dtype=np.float32)
    self._last_obs_2 = np.zeros(self._obs_size, dtype=np.float32)
    self._last_obs_3 = np.zeros(self._obs_size, dtype=np.float32)
    self._last_obs_4 = np.zeros(self._obs_size, dtype=np.float32)
    self._last_obs_5 = np.zeros(self._obs_size, dtype=np.float32)

    self._counter = 0
    self._n_substeps = n_substeps

    self._phase = np.array([0.0, np.pi])
    self._gait_freq = 1.7
    self._phase_dt = 2 * np.pi * self._gait_freq * ctrl_dt

    self.lc = lcm.LCM()
    subscription = self.lc.subscribe("TWIST_T", lcm_handle)
    # Set lc.fileno() to non-blocking mode (assuming it's a socket or similar)
    self.fd = self.lc.fileno()
    os.set_blocking(self.fd, False)  # Make the file descriptor non-blocking
    
    # Make initial pose stable
    self._init_wait_steps = 800


  def get_obs(self, model, data) -> np.ndarray:
    # Get lcm data, non-blocking with timeout=0
    rfds, wfds, efds = select.select([self.fd], [], [], 0.001)  
    if rfds:
      self.lc.handle()
                
    #linvel = data.sensor("local_linvel").data
    gyro = data.sensor("gyro").data
    acc = data.sensor("accelerometer").data
    imu_xmat = data.site_xmat[model.site("root").id].reshape(3, 3)
    gravity = imu_xmat.T @ np.array([0, 0, -1])
    joint_angles = data.qpos[7:] - self._default_angles
    joint_velocities = data.qvel[6:]
    
    phase = np.concatenate([np.cos(self._phase), np.sin(self._phase)])
    
    #twist_command = np.array([0.0, 0.0, 0.0])
   
    command = np.hstack([*vel_cmd_global])
    
    obs_n = np.hstack([
      gyro,
      gravity,
      joint_angles,
      joint_velocities,
      self._last_action,
      phase,
      command,
    ])

    # Stack history obs
    obs = np.hstack([
      obs_n,
      self._last_obs_1,
      self._last_obs_2,
      self._last_obs_3,
      self._last_obs_4,
      self._last_obs_5,
    ])

    
    self._last_obs_5 = self._last_obs_4
    self._last_obs_4 = self._last_obs_3
    self._last_obs_3 = self._last_obs_2
    self._last_obs_2 = self._last_obs_1
    self._last_obs_1 = obs_n

    return obs.astype(np.float32)

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self._counter += 1
    if self._counter % self._n_substeps == 0:
      obs = self.get_obs(model, data)
      onnx_input = {"obs": obs.reshape(1, -1)}
      onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
      self._last_action = onnx_pred.copy()
      
      data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles
      
      if self._counter < self._init_wait_steps:
        data.qpos[0:2] = np.zeros(2)
        data.qpos[2] = 0.55
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        
        data.qvel[0:6] = np.zeros(6)
        
        data.ctrl[:] = self._default_angles
      
      #print(data.ctrl[:])
      
      phase_tp1 = self._phase + self._phase_dt
      self._phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi


def load_callback(model=None, data=None):
  mujoco.set_mjcb_control(None)

  assets = {}
  mjx_env.update_assets(assets, hector_constants.ROOT_PATH / "xmls", "*.xml")
  mjx_env.update_assets(assets, hector_constants.ROOT_PATH / "xmls" / "meshes")
  model = mujoco.MjModel.from_xml_path(
      hector_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
      assets,
  )
  data = mujoco.MjData(model)

  mujoco.mj_resetDataKeyframe(model, data, 1)

  ctrl_dt = 0.02
  sim_dt = 0.002
  n_substeps = int(round(ctrl_dt / sim_dt))
  model.opt.timestep = sim_dt

  policy = OnnxController(
      policy_path=(_ONNX_DIR / 'joystick_s2_0830_1.onnx').as_posix(),
      default_angles=np.array(model.keyframe("home").qpos[7:]),
      ctrl_dt=ctrl_dt,
      n_substeps=n_substeps,
      action_scale=0.60
  )

  # Set first step control
  data.ctrl[:] = policy._default_angles
  mujoco.set_mjcb_control(policy.get_control)

  return model, data

if __name__ == "__main__":
  viewer.launch(
    loader=load_callback,
    show_left_ui=False,
    show_right_ui=False
  )
