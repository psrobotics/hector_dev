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


  def reset(self, rng: jax.Array) -> mjx_env.State:


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

    rewards = {}
    for term in self._reward_terms:
        reward_value = term.func(context)
        rewards[term.name] = reward_value # Store unscaled value for metrics
    return rewards