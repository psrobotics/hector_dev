"""Tests for the locomotion environments."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jp

from mujoco_playground._src import locomotion


class TestSuite(parameterized.TestCase):
  """Tests for the locomotion environments."""

  @parameterized.named_parameters(
      {"testcase_name": "test_can_create_HectorWBCFlatTerrain", 
       "env_name": "HectorWBCFlatTerrain"}
  )

  def test_single_environment(self, env_name: str) -> None:
    env = locomotion.load(env_name)
    state = jax.jit(env.reset)(jax.random.PRNGKey(42))
    state = jax.jit(env.step)(state, jp.zeros(env.action_size))
    self.assertIsNotNone(state)
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape, state.obs)
    obs_shape = obs_shape[0] if isinstance(obs_shape, tuple) else obs_shape
    self.assertEqual(obs_shape, env.observation_size)
    self.assertFalse(jp.isnan(state.data.qpos).any())


if __name__ == "__main__":
  absltest.main()