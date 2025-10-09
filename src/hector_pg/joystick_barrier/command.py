import jax
import jax.numpy as jp


# Sample in command space, command dim defined here
def sample_command(env, rng: jax.Array) -> jax.Array:
    min_bounds = jp.array(
        [
            env._config.lin_vel_x[0],
            env._config.lin_vel_y[0],
            env._config.ang_vel_yaw[0],
        ]
    )
    max_bounds = jp.array(
        [
            env._config.lin_vel_x[1],
            env._config.lin_vel_y[1],
            env._config.ang_vel_yaw[1],
        ]
    )
    default_values = jp.array(
        [
            0.0,
            0.0,
            0.0,
        ]
    )
    rng1, rng2 = jax.random.split(rng)
    cmd_sample = jax.random.uniform(
        rng1, shape=default_values.shape, minval=min_bounds, maxval=max_bounds
    )
    # With 10% reset all values to default
    cmd = jp.where(
        jax.random.bernoulli(rng2, p=env._config.reward_config.default_p),
        default_values,
        cmd_sample,
    )
    return cmd
