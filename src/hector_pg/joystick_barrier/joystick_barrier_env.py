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

from .config import default_config
from .command import sample_command
from .rewards import RewardManager


class JoystickBarrierEnv(hector_base.HectorEnv):
  """Track a Whole body control command."""

  def __init__(
      self,
      task: str = "flat_terrain", #"flat_terrain"
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
    # Get home qpos
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos, dtype=jp.float32)
    # Get qpos0 on joint level, first 7 are torso xyz and quaternion
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:], dtype=jp.float32)

    # Get joint range
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T # First joint is free (torso)
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    # Get joint names
    self._joint_names = [self.mj_model.joint(i).name for i in range(1, self.mj_model.njnt)]

    # Special joint indices
    hip_indices = []
    knee_indices = []
    for side in ["l", "r"]:
      for hip_j_name in consts.HIP_J_NAMES:
        hip_indices.append(self._mj_model.joint(f"{side}_{hip_j_name}").qposadr-7)
      for knee_j_name in consts.KNEE_J_NAMES:
        knee_indices.append(self._mj_model.joint(f"{side}_{knee_j_name}").qposadr-7)
    self._hip_indices = jp.array(hip_indices)
    self._knee_indices = jp.array(knee_indices)

    # Weight for joint pose cost
    self._joint_pose_weights = jp.array([
        0.75, 0.75, 0.01, 0.01, 0.01,  # left leg.
        0.75, 0.75, 0.01, 0.01, 0.01,  # right leg. # 0.5
        0.5, 0.5, 0.5, 0.5,   # left arm
        0.5, 0.5, 0.5, 0.5,   # right arm
    ])
    # Weights for injecting joint level noise
    # Hector come with 18, we lock arm (8dofs)
    qpos_noise_scale = np.zeros(18)
    hip_ids = [0, 1, 2, 5, 6, 7]
    kfe_ids = [3, 8]
    ffe_ids = [4, 9]
    arm_ids = [10, 11, 12, 13, 14, 15, 16, 17]
    qpos_noise_scale[hip_ids] = self._config.noise_config.scales.hip_pos
    qpos_noise_scale[kfe_ids] = self._config.noise_config.scales.kfe_pos
    qpos_noise_scale[ffe_ids] = self._config.noise_config.scales.ffe_pos
    qpos_noise_scale[arm_ids]  = 0.0
    self._qpos_noise_scale = jp.array(qpos_noise_scale)

    # Gemo ids
    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
    self._site_id = self._mj_model.site("root").id
    self._feet_site_id = np.array([self._mj_model.site(name).id for name in consts.FEET_SITES])
    self._feet_geom_id = np.array([self._mj_model.geom(name).id for name in consts.FEET_GEOMS])

    self._floor_geom_id = self._mj_model.geom("floor").id

    feet_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      feet_linvel_sensor_adr.append(list(range(sensor_adr, sensor_adr + sensor_dim)))
    self._feet_linvel_sensor_adr = jp.array(feet_linvel_sensor_adr)


  def reset(self, rng: jax.Array) -> mjx_env.State:
    pass


  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_gravity(data)[-1] < 0.0
    return (
        fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    )

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> mjx_env.Observation:
    state = []
    privileged_state = []

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
    
    context = []

    rewards = {}
    for term in self._reward_terms:
        reward_value = term.func(context)
        rewards[term.name] = reward_value # Store unscaled value for metrics
    return rewards