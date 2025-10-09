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


class Joystick(hector_base.HectorEnv):

    def __init__(
        self,
        task: str = "flat_terrain",  # "flat_terrain"
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        # Instantiate the reward manager
        self._reward_manager = RewardManager(self, self._config.reward_config)
        self._post_init()

    def _post_init(self) -> None:
        # Get home qpos
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos, dtype=jp.float32)
        # Get qpos0 on joint level, first 7 are torso xyz and quaternion
        self._default_pose = jp.array(
            self._mj_model.keyframe("home").qpos[7:], dtype=jp.float32
        )

        # Get joint range
        self._lowers, self._uppers = self.mj_model.jnt_range[
            1:
        ].T  # First joint is free (torso)
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        # Get joint names
        self._joint_names = [
            self.mj_model.joint(i).name for i in range(1, self.mj_model.njnt)
        ]

        # Special joint indices
        hip_indices = []
        knee_indices = []
        for side in ["l", "r"]:
            for hip_j_name in consts.HIP_J_NAMES:
                hip_indices.append(
                    self._mj_model.joint(f"{side}_{hip_j_name}").qposadr - 7
                )
            for knee_j_name in consts.KNEE_J_NAMES:
                knee_indices.append(
                    self._mj_model.joint(f"{side}_{knee_j_name}").qposadr - 7
                )
        self._hip_indices = jp.array(hip_indices)
        self._knee_indices = jp.array(knee_indices)

        # Weight for joint pose cost
        self._joint_pose_weights = jp.array(
            [
                0.75,
                0.75,
                0.01,
                0.01,
                0.01,  # left leg.
                0.75,
                0.75,
                0.01,
                0.01,
                0.01,  # right leg. # 0.5
                0.5,
                0.5,
                0.5,
                0.5,  # left arm
                0.5,
                0.5,
                0.5,
                0.5,  # right arm
            ]
        )
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
        qpos_noise_scale[arm_ids] = 0.0
        self._qpos_noise_scale = jp.array(qpos_noise_scale)

        # Gemo ids
        self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._site_id = self._mj_model.site("root").id
        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in consts.FEET_SITES]
        )
        self._feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
        )

        self._floor_geom_id = self._mj_model.geom("floor").id

        feet_linvel_sensor_adr = []
        for site in consts.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            feet_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._feet_linvel_sensor_adr = jp.array(feet_linvel_sensor_adr)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Applies configured randomizations to the initial state."""
        # Use a reference to the config for cleaner code
        cfg = self._config.reset_config
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        # Split RNG key once for all operations
        keys = jax.random.split(rng, 7)
        rng = keys[0]
        # Randomize root position (x, y)
        dxy = jax.random.uniform(
            keys[1], (2,), minval=cfg.root_pos_xy[0], maxval=cfg.root_pos_xy[1]
        )
        qpos = qpos.at[0:2].add(dxy)
        # Randomize root orientation (yaw)
        yaw = jax.random.uniform(
            keys[2], (1,), minval=cfg.root_yaw[0], maxval=cfg.root_yaw[1]
        )
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)
        # Scale all DoF positions
        qpos_scale = jax.random.uniform(
            keys[3], (18,), minval=cfg.dof_pos_scale[0], maxval=cfg.dof_pos_scale[1]
        )
        qpos = qpos.at[7:].multiply(qpos_scale)
        # Randomize root velocity
        root_vel_noise = jax.random.uniform(
            keys[4], (6,), minval=cfg.root_vel[0], maxval=cfg.root_vel[1]
        )
        qvel = qvel.at[0:6].set(root_vel_noise)
        # Randomize default pose with additive noise
        default_q_rand = self._default_pose
        dof_add_noise = jax.random.uniform(
            keys[5], (10,), minval=cfg.dof_pos_add[0], maxval=cfg.dof_pos_add[1]
        )
        default_q_rand = default_q_rand.at[0:10].add(dof_add_noise)
        # Special noise for specific joints
        idx = jp.array([9, 11, 14, 16], dtype=jp.int32)
        dof_add_special_noise = jax.random.uniform(
            keys[6],
            (idx.shape[0],),
            minval=cfg.dof_pos_add_special[0],
            maxval=cfg.dof_pos_add_special[1],
        )
        default_q_rand = default_q_rand.at[idx].add(dof_add_special_noise)

        # data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])
        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=qpos[7:],
            impl=self._config.impl,  # impl=self.mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        # Sample phase
        rng, key = jax.random.split(rng)
        gait_freq = jax.random.uniform(
            key, (1,), minval=cfg.gait_freq[0], maxval=cfg.gait_freq[1]
        )
        phase_dt = 2 * jp.pi * self.dt * gait_freq
        # Init phase set here, always a phase diff across 2 legs
        phase = jp.array([0, jp.pi])

        # Sample command
        rng, cmd_rng = jax.random.split(rng)
        cmd = sample_command(self, cmd_rng)

        # Sample push interval.
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        # Build info block to pass through
        info = {
            "rng": rng,
            "step": 0,
            "command": cmd,
            # In-step buffer
            "q_tar": jp.zeros(self.mjx_model.nu),
            "feet_pos_z": jp.zeros(2),
            "swing_peak": jp.zeros(2),
            "feet_air_time": jp.zeros(2),
            "contact": jp.zeros(2, dtype=bool),
            "first_contact": jp.zeros(2, dtype=bool),
            "last_contact": jp.zeros(2, dtype=bool),
            "desired_contact": jp.zeros(2, dtype=bool),
            "default_pose": default_q_rand,
            # Phase related
            "phase_dt": phase_dt,
            "phase": phase,
            # Push disturbanc
            "push": jp.zeros(2),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            # Obs buffer
            "obs_hist": jp.zeros(
                self._config.obs_size * self._config.obs_hist_len, dtype=jp.float32
            ),
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
        }

        # Get initial obs once reset is done
        contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)

        # Log metrics buffers
        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # Apply random push
        state.info["rng"], push1_rng, push2_rng, push3_rng = jax.random.split(
            state.info["rng"], 4
        )
        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= (
            jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
        )
        push *= self._config.push_config.enable
        qvel = state.data.qvel
        qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        # Get current action and step
        q_tar = self._default_pose + action * self._config.action_scale
        state.info["q_tar"] = q_tar
        # Env step
        data = mjx_env.step(self.mjx_model, state.data, q_tar, self.n_substeps)

        # Foot contact
        is_contact_gemo = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        is_contact_force = (
            jp.array(
                [
                    jp.abs(
                        mjx_env.get_sensor_data(
                            self.mj_model, data, consts.FEET_FORCE_SENSOR[0]
                        )[2]
                    ),
                    jp.abs(
                        mjx_env.get_sensor_data(
                            self.mj_model, data, consts.FEET_FORCE_SENSOR[1]
                        )[2]
                    ),
                ]
            )
            > self._config.reward_config.feet_f_contact
        )
        contact = jp.logical_and(is_contact_gemo, is_contact_force)
        state.info["contact"] = contact
        first_contact = jp.logical_and(
            jp.logical_not(state.info["last_contact"]), contact
        )
        state.info["first_contact"] = first_contact

        # Foot pos
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["feet_pos_z"] = p_fz  # dim = 2

        air_time_prev = state.info["feet_air_time"]
        state.info["feet_air_time"] = jp.where(contact, 0.0, air_time_prev + self.dt)

        state.info["swing_peak"] = jp.where(
            contact, 0.0, jp.maximum(state.info["swing_peak"], p_fz)
        )

        state.info["last_contact"] = contact

        state.info["step"] += 1
        state.info["push_step"] += 1
        state.info["push"] = push

        phase_tp1 = state.info["phase"] + state.info["phase_dt"]
        state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi  # 2d
        state.info["desired_contact"] = (
            gait.get_rz_phase(
                state.info["phase"],
                self._config.reward_config.max_foot_height,
                self._config.reward_config.airtime,
            )
            <= 5e-3
        )
        # Update action buffer
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action

        # Get observation and termination flags
        obs = self._get_obs(data, state.info, contact)
        done = self._get_termination(data)

        # Update history obs
        obs_n = obs["state"][: self._config.obs_size]
        state.info["obs_hist"] = jp.concatenate(
            [obs_n, state.info["obs_hist"][: -self._config.obs_size]]
        )

        # Resample each 500 steps
        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
        state.info["command"] = jp.where(
            state.info["step"] > self._config.resample_step_interval,
            sample_command(self, cmd_rng),
            state.info["command"],
        )
        state.info["step"] = jp.where(
            done | (state.info["step"] > self._config.resample_step_interval),
            0,
            state.info["step"],
        )

        done = done.astype(jp.float32)
        total_rewards, unscaled_rewards = self._get_reward(
            data, action, state.info, done
        )
        reward = jp.clip(total_rewards * self.dt, -1e5, 1e5)
        # Log unscaled rewards
        for k, v in unscaled_rewards.items():
            state.metrics[f"reward/{k}"] = v
            
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_gravity(data)[-1] < 0.0
        return fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
    ) -> mjx_env.Observation:
        state = []
        privileged_state = []

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
        phase_obs = jp.concatenate([cos, sin])

        linvel = self.get_local_linvel(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_linvel = (
            linvel
            + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.linvel
        )

        state_n = jp.hstack(
            [
                noisy_gyro,  # 3
                noisy_gravity,  # 3
                noisy_joint_angles - info["default_pose"],  # 18
                noisy_joint_vel,  # 18
                info["last_act"],  # 18
                phase_obs,  # 4
                info["command"],  # 3
            ]
        )
        # Stack history obs buffer
        state = jp.hstack(
            [
                state_n,
                info["obs_hist"],
            ]
        )

        accelerometer = self.get_accelerometer(data)
        global_angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._feet_linvel_sensor_adr].ravel()
        root_height = data.qpos[2]

        privileged_state = jp.hstack(
            [
                state,
                gyro,  # 3
                accelerometer,  # 3
                gravity,  # 3
                linvel,  # 3
                global_angvel,  # 3
                joint_angles - info["default_pose"],
                joint_vel,
                root_height,  # 12
                data.actuator_force,  # 18
                contact,  # 2
                feet_vel,  # 4*3
                info["feet_air_time"],  # 2
            ]
        )

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        done: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:

        context = {
            # --- Base data ---
            "data": data,
            "action": action,
            "info": info,
            "done": done,
            # --- Pre-computed values ---
            "command": info["command"],
            "contact": info["contact"],
            "first_contact": info["first_contact"],
            "desired_contact": info["desired_contact"],
            "local_linvel": self.get_local_linvel(data),
            "global_linvel": self.get_global_linvel(data),
            "global_angvel": self.get_global_angvel(data),
            "body_height": data.qpos[2],
            "gravity": self.get_gravity(data),
            "gyro": self.get_gyro(data),
            "q": data.qpos[7:],
            "qarm": data.qpos[17:25],
            "qvel": data.qvel[6:],
            "qacc": data.qacc[6:],
            "act_frc": data.actuator_force,
            "last_act": info["last_act"],
            "last_last_act": info["last_last_act"],
            "phase": info["phase"],
            "feet_air_time": info["feet_air_time"],
            "p_fz": info["feet_pos_z"],
            "zaxis_fz": self.get_feet_zaxis(data),
            "default_pose": info["default_pose"],
            "joint_pose_weights": self._joint_pose_weights,
            "soft_lowers": self._soft_lowers,
            "soft_uppers": self._soft_uppers,
            "hip_indices": self._hip_indices,
            "knee_indices": self._knee_indices,
            "feet_linvel_sensor_adr": self._feet_linvel_sensor_adr,
            "feet_site_id": self._feet_site_id,
            "airtime": self._config.reward_config.airtime,
            "max_foot_height": self._config.reward_config.max_foot_height,
            "max_fz": self._config.reward_config.max_contact_force,
            "tar_body_height": self._config.body_height_default,
            "f_dist_range": self._config.f_dist_range,
            "tar_body_height": self._config.body_height_default,
        }

        total_scaled_reward, unscaled_rewards = self._reward_manager.calculate_rewards(
            context
        )

        return total_scaled_reward, unscaled_rewards
