"""WBC task for Hector."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp

from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from hector_pg import base as hector_base
from hector_pg import constants as consts
from hector_pg.utils import ankle_decouple

from dataclasses import dataclass
from typing import Callable

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=1000,
      action_repeat=1,
      action_scale=0.75,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      # OBS size
      obs_size = 67,
      obs_hist_len = 20,
      # Noise scales
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              hip_pos=0.03,  # rad
              kfe_pos=0.05,
              ffe_pos=0.08,
              faa_pos=0.03,
              joint_vel=1.5,  # rad/s
              gravity=0.05,
              linvel=0.1,
              gyro=0.2,  # angvel.
              acc=0.2,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # --- Tracking related rewards ---
              tracking_lin_vel=2.0,
              tracking_ang_vel=1.5,
              #tracking_vel_hard=0.0,
              #tracking_body_height=0.0,
              #tracking_body_euler=0.0,
              #tracking_arm=0.0,
              # --- Base related rewards ---
              lin_vel_z=-0.15,
              ang_vel_xy=-0.25,#-0.25,
              orientation=1.0,
              # --- Energy related rewards ---
              energy=-0.0,
              smoothness=-0.0,
              contact_force=-0.0,
              #dof_acc = -0.0, #-1e-7,
              #dof_vel = -0.0, #-1e-4,
              # --- Feet related rewards ---
              #feet_air_time=2.0,
              feet_height=2.0,
              feet_slip=-0.75,
              undesired_contact=-3.0,
              feet_upright=-0.25,
              feet_dist=-0.0,#-5e4,
              # --- Other rewards ---
              alive=0.5,
              termination=-1.0,
              #stand_still=-0.0, # -1.0
              # --- Pose related rewards ---
              #joint_deviation_knee=-0.0,
              joint_deviation_hip=-0.25,
              dof_pos_limits=-0.25,
              pose=-0.5,
          ),
          max_foot_height=0.10,
          max_contact_force=250.0,
          # Force threshold that holds as contact
          feet_f_contact = 5.0,
          # Desired airtime within phase (1.0 scale)
          airtime = 0.45, #0.45
          # In what precentage control will be ruleout
          default_p = 0.1,
      ),
      push_config=config_dict.create(
          # Disable first to get a init policy
          enable=True,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 2.0],
      ),
      # Command sampling ranges
      lin_vel_x=[-1.0, 1.0],
      lin_vel_y=[-1.0, 1.0],
      ang_vel_yaw=[-1.0, 1.0],
      # Feet distance min  max
      f_dist_range=[0.08, 0.4],
      # Default body height
      body_height_default=0.55,

      delay = config_dict.create(
          min_delay_step=0,
          max_delay_step=10,
          hist_len=10,
      ),
  
      impl="jax", # "jax" or "warp"
      nconmax=8 * 8192,
      njmax=60,
  )

@dataclass
class RewardTerm:
  """A container for a single reward component."""
  name: str
  scale: float
  func: Callable[..., jax.Array]
  
class Joystick(hector_base.HectorEnv):
  """Track a Whole body control command."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    # First 7 are xyz and rpy quaternion
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    # Note: First joint is freejoint, root(torso) joint
    # Also get joint range
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor
    
    # Print joint limits
    joint_names = [self.mj_model.joint(i).name for i in range(1, self.mj_model.njnt)]
    for name, low, high in zip(joint_names, self._lowers, self._uppers): print(f"{name}: {low:.2f} to {high:.2f}")

    hip_indices = []
    # To keep hip yaw, roll angle near default angle here
    hip_joint_names = ["hip_yaw", "hip_roll"]
    for side in ["l", "r"]:
      for joint_name in hip_joint_names:
        hip_indices.append(
            self._mj_model.joint(f"{side}_{joint_name}").qposadr - 7
        )
    self._hip_indices = jp.array(hip_indices)

    knee_indices = []
    for side in ["l", "r"]:
      knee_indices.append(self._mj_model.joint(f"{side}_knee").qposadr - 7)
    self._knee_indices = jp.array(knee_indices)

    # fmt: off
    self._weights = jp.array([
        0.7, 0.7, 0.01, 0.01, 0.01,  # left leg.
        0.7, 0.7, 0.01, 0.01, 0.01,  # right leg. # 0.5
        0.5, 0.5, 0.5, 0.5,   # left arm
        0.5, 0.5, 0.5, 0.5,   # right arm
    ])
    # fmt: on

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
    # Torso
    self._site_id = self._mj_model.site("root").id

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    # Hector come with 18, we lock arm (8dofs)
    qpos_noise_scale = np.zeros(18)
    hip_ids = [0, 1, 2, 5, 6, 7]
    kfe_ids = [3, 8]
    ffe_ids = [4, 9]
    arm_ids = [10, 11, 12, 13, 14, 15, 16, 17]
    #faa_ids = [5, 11]
    qpos_noise_scale[hip_ids] = self._config.noise_config.scales.hip_pos
    qpos_noise_scale[kfe_ids] = self._config.noise_config.scales.kfe_pos
    qpos_noise_scale[ffe_ids] = self._config.noise_config.scales.ffe_pos
    qpos_noise_scale[arm_ids]  = 0.0
    #qpos_noise_scale[faa_ids] = self._config.noise_config.scales.faa_pos
    self._qpos_noise_scale = jp.array(qpos_noise_scale)
    
    # Init reward terms
    reward_function_mapping = {
        # --- Tracking rewards ---
        'tracking_lin_vel': self._reward_tracking_lin_vel,
        'tracking_ang_vel': self._reward_tracking_ang_vel,
        #'tracking_vel_hard': self._reward_tracking_vel_hard,
        #'tracking_body_height': self._reward_tracking_body_height,
        # --- Stay balanced ---
        'lin_vel_z': self._cost_lin_vel_z,
        'ang_vel_xy': self._cost_ang_vel_xy,
        'orientation': self._reward_base_orientation,
        # --- Energy terms ---
        'energy': self._cost_energy,
        'smoothness': self._cost_smoothness,
        #'dof_acc': self._cost_dof_acc,
        #'dof_vel': self._cost_dof_vel,
        'contact_force': self._cost_contact_force,
        # --- Gait shaping ---
        'feet_height': self._reward_feet_height,
        #'feet_air_time': self._reward_feet_air_time,
        'feet_slip': self._cost_feet_slip,
        'undesired_contact': self._cost_undesired_contact_phase,
        'feet_upright': self._cost_feet_upright,
        'feet_dist': self._cost_feet_dist,
        # --- Alive ---
        'alive': self._reward_alive,
        'termination': self._cost_termination,
        #'stand_still': self._cost_stand_still,
        # --- Others ---
        'dof_pos_limits': self._cost_joint_pos_limits,
        'pose': self._cost_pose,
      # Add other rewards here as we create them
    }

    # Build the list of active reward terms from the config
    self._reward_terms = []
    for name, scale in self._config.reward_config.scales.items():
        if name in reward_function_mapping and scale != 0.0:
            self._reward_terms.append(
                RewardTerm(name=name, scale=scale, func=reward_function_mapping[name])
            )
            

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # qpos[7:]=*U(0.5, 1.5)
    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:].set(
        qpos[7:] * jax.random.uniform(key, (18,), minval=0.5, maxval=1.5)
    )

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )
    
    #data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])
    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:],
        impl=self._config.impl, #impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    # Phase, freq=U(1.7, 2.0)
    rng, key = jax.random.split(rng)
    gait_freq = jax.random.uniform(key, (1,), minval=1.7, maxval=2.0)
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    # Init phase set here, always a phase diff across 2 legs
    phase = jp.array([0, jp.pi])

    rng, cmd_rng = jax.random.split(rng)
    cmd = self.sample_command(cmd_rng)

    # Sample push interval.
    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,

        "q_tar": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "desired_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        "feet_pos_z": jp.zeros(2),

        # Phase related.
        "phase_dt": phase_dt,
        "phase": phase,
        # Push related.
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
        # Past obs
        "obs_hist": jp.zeros(self._config.obs_size*self._config.obs_hist_len,
                              dtype=jp.float32),
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
      
        # OBS to train forward dynamics
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())
    metrics["p_fz"] = jp.zeros(())

    contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    obs = self._get_obs(data, info, contact)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)


  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state.info["rng"], push1_rng, push2_rng = jax.random.split(
        state.info["rng"], 3
    )
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=self._config.push_config.magnitude_range[0],
        maxval=self._config.push_config.magnitude_range[1],
    )
    push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
    push *= (
        jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
        == 0
    )
    push *= self._config.push_config.enable
    qvel = state.data.qvel
    qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
    data = state.data.replace(qvel=qvel)
    state = state.replace(data=data)
    
    # Get position level joint control
    q_tar = self._default_pose + action * self._config.action_scale  

    data = mjx_env.step(
        self.mjx_model, state.data, q_tar, self.n_substeps
    )
    state.info["q_tar"] = q_tar

    # Gemo based contact event
    contact_gemo = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    # Force based contact event
    contact_force = jp.array([
      jp.abs(mjx_env.get_sensor_data(self.mj_model, data, "left_foot_force")[2]),
      jp.abs(mjx_env.get_sensor_data(self.mj_model, data, "right_foot_force")[2])
    ]) > self._config.reward_config.feet_f_contact
    # Filter out false contacts
    contact = contact_gemo & contact_force
    #contact=contact_gemo
    
    last_contact = state.info["last_contact"] 
    air_time_prev = state.info["feet_air_time"] 
    # Touchdown = rising edge of contact
    first_contact = jp.logical_and(jp.logical_not(last_contact), contact)

    obs = self._get_obs(data, state.info, contact)
    done = self._get_termination(data)

    # Get rewards and scale
    unscaled_rewards = self._get_reward(
        data, action, state.info, first_contact, contact, done
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in unscaled_rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, -1e5, 1e5)

    state.info["push"] = push
    state.info["step"] += 1
    state.info["push_step"] += 1
    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    # Phase is 2d
    state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    # Target stance contact, match giat_rz
    state.info["desired_contact"] = gait.get_rz_phase(state.info["phase"],
                                                      self._config.reward_config.max_foot_height,
                                                      self._config.reward_config.airtime) <= 5e-3
    #jp.sin(state.info["phase"]) <= 0    
    
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
  
    # Update history obs
    obs_n = obs["state"][: self._config.obs_size] 
    state.info["obs_hist"] = jp.concatenate([obs_n,
                                             state.info["obs_hist"][:-self._config.obs_size]])

    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    
    # Sample twice
    state.info["command"] = jp.where(
        state.info["step"] > 500,
        self.sample_command(cmd_rng),
        state.info["command"],
    )
    
    state.info["step"] = jp.where(
        done | (state.info["step"] > 500),
        0,
        state.info["step"],
    )
    
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["feet_pos_z"] = p_fz # dim = 2
    state.info["feet_air_time"] = jp.where(contact, 0.0, air_time_prev + self.dt)
    state.info["swing_peak"]    = jp.where(contact,
                                           0.0,
                                           jp.maximum(state.info["swing_peak"], p_fz))
    state.info["last_contact"]  = contact

    # Store scaled rewards for logging
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])
    state.metrics["p_fz"] = jp.mean(p_fz)

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_gravity(data)[-1] < 0.0
    return (
        fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    )

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> mjx_env.Observation:
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    acc = self.get_accelerometer(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_acc = (
        acc
        + (2 * jax.random.uniform(noise_rng, shape=acc.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.acc
    )
    
    gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._qpos_noise_scale
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase = jp.concatenate([cos, sin])

    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    state_n = jp.hstack([
      noisy_gyro,# 3
      #noisy_acc,
      noisy_gravity, # 3
      noisy_joint_angles - self._default_pose, # 18
      noisy_joint_vel, # 18
      info["last_act"], # 18
      phase, # 4
      info["command"], # 3
    ])
    # Stack history obs
    state = jp.hstack([
      state_n,
      info["obs_hist"],
    ])

    accelerometer = self.get_accelerometer(data)
    global_angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    root_height = data.qpos[2]

    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        global_angvel,  # 3
        joint_angles - self._default_pose,
        joint_vel,
        root_height,  # 12
        data.actuator_force,  # 18
        contact,  # 2
        feet_vel,  # 4*3
        info["feet_air_time"],  # 2
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      first_contact: jax.Array,
      contact: jax.Array,
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    context = {
      'data': data,
      'action': action,
      'last_act': info['last_act'],
      'last_last_act': info['last_last_act'],
      'info': info,
      'command': info['command'],
      'phase': info['phase'],
      'contact': contact,
      'first_contact': first_contact,
      'desired_contact': info['desired_contact'],
      'done': done,
      'local_linvel': self.get_local_linvel(data),
      'global_linvel': self.get_global_linvel(data),
      'gyro': self.get_gyro(data),
      'global_angvel': self.get_global_angvel(data),
      'gravity': self.get_gravity(data),
      'body_height': data.qpos[2],
      "q": data.qpos[7:25],
      'qarm': data.qpos[17:25],
      'qvel': data.qvel[6:],
      'qacc': data.qacc[6:],
      'act_frc': data.actuator_force,
      'torso_zaxis': self.get_gravity(data),
      'feet_air_time': info["feet_air_time"],
      'p_fz': info['feet_pos_z'],
      'zaxis_fz': self.get_feet_zaxis(data),
      
      'airtime': self._config.reward_config.airtime,
      'max_foot_height': self._config.reward_config.max_foot_height,
      'max_fz': self._config.reward_config.max_contact_force,
      'tar_body_height': self._config.body_height_default,
      'f_dist_range': self._config.f_dist_range,
    }
    rewards = {}
    for term in self._reward_terms:
        reward_value = term.func(context)
        rewards[term.name] = reward_value # Store unscaled value for metrics
    return rewards


  # Tracking rewards.
  def _reward_tracking_lin_vel(self, context: Dict[str, Any]) -> jax.Array:
    commands = context['command']
    local_vel = context['local_linvel']
    sigma = 0.25
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / sigma)

  # Yaw tracking reward
  def _reward_tracking_ang_vel(self, context: Dict[str, Any]) -> jax.Array:
    commands = context['command']
    ang_vel = context['gyro']
    sigma = 0.25
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / sigma)
  
  # Base-related rewards and penalties
  def _cost_lin_vel_z(self, context: Dict[str, Any]) -> jax.Array:
    return jp.square(context['global_linvel'][2])

  def _cost_ang_vel_xy(self, context: Dict[str, Any]) -> jax.Array:
    return jp.sum(jp.square(context['global_angvel'][:2]))

  def _reward_base_orientation(self, context: Dict[str, Any]) -> jax.Array:
    torso_zaxis = context['torso_zaxis']
    sigma = 0.25
    err = jp.sum(jp.square(torso_zaxis[:2]))
    return jp.exp(-err/sigma)

  # Energy related rewards.
  def _cost_energy(self, context: Dict[str, Any]) -> jax.Array:
    qvel = context['qvel']
    qfrc = context['act_frc']
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc))
  
  def _cost_smoothness(self, context: Dict[str, Any]) -> jax.Array:
    act = context['action']
    last_act = context['last_act']
    last_last_act = context['last_last_act']
    c1 = jp.sum(jp.square(act - last_act))
    c2 = jp.sum(jp.square(act - 2*last_act + last_last_act))
    return (c1+c2)
  
  def _cost_dof_acc(self, context: Dict[str, Any]) -> jax.Array:
    return jp.sum(jp.square(context['qacc']))

  def _cost_dof_vel(self, context: Dict[str, Any]) -> jax.Array:
    return jp.sum(jp.square(context['qvel']))

  # Others
  def _cost_joint_pos_limits(self, context: Dict[str, Any]) -> jax.Array:
    qpos = context['q']
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _cost_termination(self, context: Dict[str, Any]) -> jax.Array:
    return context['done']

  def _reward_alive(self, context: Dict[str, Any]) -> jax.Array:
    return jp.array(1.0)

  # Pose-related rewards.
  def _cost_joint_deviation_hip(self, context: Dict[str, Any]) -> jax.Array:
    qpos = context['q']
    cost = jp.sum(jp.abs(qpos[self._hip_indices] - self._default_pose[self._hip_indices]))
    cost *= jp.abs(context['command'][1]) > 0.1 # Maskout if there is large vy command
    return cost

  def _cost_joint_deviation_knee(self, context: Dict[str, Any]) -> jax.Array:
    qpos = context['q']
    err = qpos[self._knee_indices] - self._default_pose[self._knee_indices]
    return jp.sum(jp.abs(err))
    
  def _cost_contact_force(self, context: Dict[str, Any]) -> jax.Array:
    data = context['data']
    max_fz = context['max_fz']
    l_f = mjx_env.get_sensor_data(self.mj_model, data, "left_foot_force")
    r_f = mjx_env.get_sensor_data(self.mj_model, data, "right_foot_force")
    l_fz = l_f[2]
    r_fz = r_f[2]
    return jp.clip(jp.abs(l_fz)+jp.abs(r_fz), 0.0, 200.0)

  def _cost_pose(self, context: Dict[str, Any]) -> jax.Array:
    qpos = context['q']
    return jp.sum(jp.square(qpos - self._default_pose) * self._weights)

  # Feet related rewards.
  def _cost_feet_slip(self, context: Dict[str, Any]) -> jax.Array:
    data = context['data']
    contact = context['contact']
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]  # (2, 3)
    v_tan = jp.linalg.norm(feet_vel[..., :2], axis=-1)        # (2,)
    # Penalize slip only when that foot is in contact
    return jp.sum(jp.where(contact, v_tan, 0.0))

  def _reward_feet_height(self, context: Dict[str, Any]) -> jax.Array:
    sigma = 0.0004
    rz = gait.get_rz_phase(context['phase'],
                           swing_height=context['max_foot_height'],
                           airtime=context['airtime'])    
    err = jp.clip(rz - context['p_fz'], min=0.0)
    squared_err = jp.square(err)
    rew_per_foot = jp.exp(-squared_err/sigma)
    # Rule out zero commands, stand still
    cmd_norm = jp.linalg.norm(context['command'][0:3])
    rew_swing = rew_per_foot * jp.logical_not(context['contact']) * (cmd_norm>0.1)
    return jp.sum(rew_swing)
  
  def _cost_feet_upright(self, context: Dict[str, Any]) -> jax.Array:
    z_fz = context['zaxis_fz']
    c_l = jp.sum(jp.square(z_fz[0])) # Only care about x axis projection
    c_r = jp.sum(jp.square(z_fz[3]))
    return c_l+c_r
  
  def _cost_feet_dist(self, context: Dict[str, Any]) -> jax.Array:
    p_f = context['data'].site_xpos[self._feet_site_id]
    dmin, dmax = context['f_dist_range'] # min max of target feet distance
    dist = jp.linalg.norm(p_f[0,:2]-p_f[1,:2])
    under = jp.maximum(dmin - dist, 0.0)
    over  = jp.maximum(dist - dmax, 0.0)
    violation = under + over
    return violation*violation
  
  def _cost_stand_still(self, context: Dict[str, Any]) -> jax.Array:
    commands = context['command']
    qpos = context['q']
    cmd_norm_twist = jp.linalg.norm(commands[0:3])
    cmd_norm_track = jp.linalg.norm(commands[3:7]-jp.array([context['tar_body_height'], 0.0, 0.0, 0.0]))
    enable = (cmd_norm_twist<0.1) & (cmd_norm_track<0.25)
    return jp.sum(jp.abs(qpos[0:10] - self._default_pose[0:10])) * enable

  def _cost_undesired_contact_phase(self, context: Dict[str, Any]) -> jax.Array:
    contact = context['contact']
    desired_contact = context['desired_contact']
    mismatch = jp.not_equal(contact, desired_contact) 
    return jp.sum(mismatch.astype(jp.float32))
  
  # Sample in command space, command dim defined here
  def sample_command(self, rng: jax.Array) -> jax.Array:
    min_bounds = jp.array([
      self._config.lin_vel_x[0],        
      self._config.lin_vel_y[0],   
      self._config.ang_vel_yaw[0], 
    ])
    max_bounds = jp.array([
      self._config.lin_vel_x[1],        
      self._config.lin_vel_y[1],     
      self._config.ang_vel_yaw[1],  
    ])
    default_values = jp.array([
      0.0,
      0.0,
      0.0,
    ])
    rng1, rng2 = jax.random.split(rng)
    cmd_sample = jax.random.uniform(
        rng1,
        shape=default_values.shape,
        minval=min_bounds,
        maxval=max_bounds
    )
    # With 10% reset all values to default
    cmd = jp.where(
        jax.random.bernoulli(rng2, p=self._config.reward_config.default_p),
        default_values,
        cmd_sample
    )
    return cmd
