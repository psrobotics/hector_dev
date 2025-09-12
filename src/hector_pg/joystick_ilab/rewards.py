from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp

from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding


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

def _cost_orientation(self, context: Dict[str, Any]) -> jax.Array:
    torso_zaxis = context['torso_zaxis']
    return jp.sum(jp.square(torso_zaxis[:2]))

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
    cost = jp.sum(jp.abs(qpos[self._hip_indices] - context['default_pose'][self._hip_indices]))
    cost *= jp.abs(context['command'][1]) > 0.1 # Maskout if there is large vy command
    return cost

def _cost_joint_deviation_knee(self, context: Dict[str, Any]) -> jax.Array:
    qpos = context['q']
    err = qpos[self._knee_indices] - context['default_pose'][self._knee_indices]
    return jp.sum(jp.abs(err))

def _cost_contact_force(self, context: Dict[str, Any]) -> jax.Array:
    data = context['data']
    max_fz = context['max_fz']
    l_f = mjx_env.get_sensor_data(self.mjx_model, data, "left_foot_force")
    r_f = mjx_env.get_sensor_data(self.mjx_model, data, "right_foot_force")
    l_fz = l_f[2]
    r_fz = r_f[2]
    return jp.clip(jp.abs(l_fz)+jp.abs(r_fz), 0.0, 200.0)

def _cost_pose(self, context: Dict[str, Any]) -> jax.Array:
    qpos = context['q']
    return jp.sum(jp.square(qpos - context['default_pose']) * self._weights)

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
    return jp.sum(jp.abs(qpos[0:10] - context['default_pose'][0:10])) * enable

def _cost_undesired_contact_phase(self, context: Dict[str, Any]) -> jax.Array:
    contact = context['contact']
    desired_contact = context['desired_contact']
    mismatch = jp.not_equal(contact, desired_contact) 
    return jp.sum(mismatch.astype(jp.float32))
