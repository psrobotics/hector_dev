
"""Domain randomization for Hector."""

import jax
from mujoco import mjx
import numpy as np

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1
ANKLE_JOINT_IDS = np.array([[4, 9]])

def domain_randomize(model: mjx.Model, rng: jax.Array):
  @jax.vmap
  def rand_dynamics(rng):
    # Floor friction: =U(0.4, 1.0).
    rng, key = jax.random.split(rng)
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
        jax.random.uniform(key, minval=0.4, maxval=1.0)
    )

    # Scale static friction: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
        key, shape=(18,), minval=0.9, maxval=1.1
    )
    dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

    # Scale armature: 
    rng, key = jax.random.split(rng)
    armature = model.dof_armature[6:] * jax.random.uniform(
        key, shape=(18,), minval=0.9, maxval=1.1
    )
    dof_armature = model.dof_armature.at[6:].set(armature)

    # Scale all link masses: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(model.nbody,), minval=0.9, maxval=1.1
    )
    body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

    # Add mass to torso: +U(-1.0, 1.0).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(key, minval=-1.0, maxval=2.0)
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + dmass
    )

    # Jitter qpos0: +U(-0.05, 0.05).
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[7:].set(
        qpos0[7:]
        + jax.random.uniform(key, shape=(18,), minval=-0.05, maxval=0.05)
    )

    # Joint stiffness: *U(0.8, 1.2).
    rng, key = jax.random.split(rng)
    kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
        key, (model.nu,), minval=0.8, maxval=1.2
    )
    actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
    actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

    # Joint damping: *U(0.8, 1.2).
    rng, key = jax.random.split(rng)
    kd = model.dof_damping[6:] * jax.random.uniform(
        key, (18,), minval=0.8, maxval=1.2
    )
    dof_damping = model.dof_damping.at[6:].set(kd)

    rng, key = jax.random.split(rng)
    kd = model.dof_damping[ANKLE_JOINT_IDS] * jax.random.uniform(
        key, (2,), minval=0.5, maxval=2.0
    )
    dof_damping = model.dof_damping.at[ANKLE_JOINT_IDS].set(kd)
    
    return (
        geom_friction,
        dof_frictionloss,
        dof_armature,
        body_mass,
        qpos0,
        actuator_gainprm,
        actuator_biasprm,
        dof_damping,
    )

  (
      friction,
      frictionloss,
      armature,
      body_mass,
      qpos0,
      actuator_gainprm,
      actuator_biasprm,
      dof_damping,
  ) = rand_dynamics(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_mass": 0,
      "qpos0": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
      "dof_damping": 0,
  })

  model = model.tree_replace({
      "geom_friction": friction,
      "dof_frictionloss": frictionloss,
      "dof_armature": armature,
      "body_mass": body_mass,
      "qpos0": qpos0,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
      "dof_damping": dof_damping,
  })

  return model, in_axes
