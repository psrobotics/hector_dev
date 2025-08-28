import jax
import jax.numpy as jp

# Indices (2 legs × 5 joints) + 8 arm joints = 18 total
HIP_IDX   = jp.array([0, 1, 2, 5, 6, 7])
KNEE_IDX  = jp.array([3, 8])
ANKLE_IDX = jp.array([4, 9])
ARM_IDX   = jp.arange(10, 18)
ARM_SPEC  = jp.array([13, 17])   # joints with extra gear_ratio_arm

MOT_DIR = jp.array([1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1,
                    1, 1, 1, 1,  # ori 1 1 -1 1 we'll not handle joint dir here
                    1, 1, 1, 1])
GEAR_RATIO_KNEE = 2.0
GEAR_RATIO_ARM = 1.417
KNEE_OFFSET = -2.38

# motor-space PD for [m_knee, m_ankle] (tune as needed)
Kp_m = jp.array([35.0, 35.0], dtype=jp.float32)
Kd_m = jp.array([ 1.5,  1.5], dtype=jp.float32)

# sim position-actuator PD (must match your MJCF)
# you only gave ankle gains; we’ll start knees = ankles (adjust if different)
ANKLE_KP, ANKLE_KD = 35.0, 1.5
KNEE_KP,  KNEE_KD  = 35.0, 1.5  

NJ = 18  # total joints you control (7..7+18 in qpos)
Kp_j = jp.full((NJ,), ANKLE_KP, dtype=jp.float32)
Kd_j = jp.full((NJ,), ANKLE_KD, dtype=jp.float32)
Kp_j = Kp_j.at[KNEE_IDX].set(KNEE_KP)
Kd_j = Kd_j.at[KNEE_IDX].set(KNEE_KD)
invKp_j = 1.0 / jp.maximum(Kp_j, 1e-6)

# pair indices (L/R rows: [knee, ankle])
pairs = jp.stack([KNEE_IDX, ANKLE_IDX], axis=1)  # shape (2,2), int32

# coupling map (motor = A @ joint + b) for knee/ankle
r = GEAR_RATIO_KNEE
ko = KNEE_OFFSET
A  = jp.array([[r, 0.0],
               [1.0, 1.0]], dtype=jp.float32)
b  = jp.array([(1.0 - r) * ko,
                -ko], dtype=jp.float32)
AT = A.T


def obs_ik_qdq(policy_q,
               policy_dq,
               joint_dir: jax.Array = MOT_DIR,
               knee_offset: float = KNEE_OFFSET,
               gear_ratio_knee: float = GEAR_RATIO_KNEE,
               gear_ratio_arm: float = GEAR_RATIO_ARM,
               )->jax.Array:
    '''Decouple obs from joint(policy) space to motor(low-level) space'''
    low_q  = jp.zeros_like(policy_q)
    low_dq = jp.zeros_like(policy_dq)

    # hips: direct sign
    low_q  = low_q.at[HIP_IDX].set(joint_dir[HIP_IDX] * policy_q[HIP_IDX])
    low_dq = low_dq.at[HIP_IDX].set(joint_dir[HIP_IDX] * policy_dq[HIP_IDX])

    # knee: (q - off)*ratio + off, then sign
    qd3 = policy_q[KNEE_IDX]; dd3 = policy_dq[KNEE_IDX]
    q3_raw  = (qd3 - knee_offset) * gear_ratio_knee + knee_offset
    dq3_raw = dd3 * gear_ratio_knee
    low_q  = low_q.at[KNEE_IDX].set(joint_dir[KNEE_IDX] * q3_raw)
    low_dq = low_dq.at[KNEE_IDX].set(joint_dir[KNEE_IDX] * dq3_raw)

    # ankle: depends on knee
    qd4 = policy_q[ANKLE_IDX]; dd4 = policy_dq[ANKLE_IDX]
    q4_raw  = qd4 + qd3 - knee_offset
    dq4_raw = dd4 + dd3
    low_q  = low_q.at[ANKLE_IDX].set(joint_dir[ANKLE_IDX] * q4_raw)
    low_dq = low_dq.at[ANKLE_IDX].set(joint_dir[ANKLE_IDX] * dq4_raw)

    # arms: direct sign
    low_q  = low_q.at[ARM_IDX].set(joint_dir[ARM_IDX] * policy_q[ARM_IDX])
    low_dq = low_dq.at[ARM_IDX].set(joint_dir[ARM_IDX] * policy_dq[ARM_IDX])

    # special arm joints: apply gear ratio
    low_q  = low_q.at[ARM_SPEC].set(joint_dir[ARM_SPEC] * policy_q[ARM_SPEC] * gear_ratio_arm)
    low_dq = low_dq.at[ARM_SPEC].set(joint_dir[ARM_SPEC] * policy_dq[ARM_SPEC] * gear_ratio_arm)

    return jp.hstack([low_q, low_dq])


def act_fk_qdq(low_q,
               low_dq,
               joint_dir: jax.Array = MOT_DIR,
               knee_offset: float = KNEE_OFFSET,
               gear_ratio_knee: float = GEAR_RATIO_KNEE,
               gear_ratio_arm: float = GEAR_RATIO_ARM,
               )->jax.Array:
    '''Couple act from motor(low-level) space to joint(policy) space'''
    # 1) undo sign to get motor coords (what IK used before sign)
    m_q  = joint_dir * low_q
    m_dq = joint_dir * low_dq

    policy_q  = jp.zeros_like(low_q)
    policy_dq = jp.zeros_like(low_dq)

    # hips: just sign-corrected values
    policy_q  = policy_q.at[HIP_IDX].set(m_q[HIP_IDX])
    policy_dq = policy_dq.at[HIP_IDX].set(m_dq[HIP_IDX])

    # knee: invert offset+ratio
    q3 = (m_q[KNEE_IDX] - knee_offset) / gear_ratio_knee + knee_offset
    dq3 = m_dq[KNEE_IDX] / gear_ratio_knee
    policy_q  = policy_q.at[KNEE_IDX].set(q3)
    policy_dq = policy_dq.at[KNEE_IDX].set(dq3)

    # ankle: invert coupling with knee
    policy_q  = policy_q.at[ANKLE_IDX].set(m_q[ANKLE_IDX] - q3 + knee_offset)
    policy_dq = policy_dq.at[ANKLE_IDX].set(m_dq[ANKLE_IDX] - dq3)

    # arms: sign-corrected
    policy_q  = policy_q.at[ARM_IDX].set(m_q[ARM_IDX])
    policy_dq = policy_dq.at[ARM_IDX].set(m_dq[ARM_IDX])

    # special arm joints: divide the gear ratio
    policy_q  = policy_q.at[ARM_SPEC].set(m_q[ARM_SPEC] / gear_ratio_arm)
    policy_dq = policy_dq.at[ARM_SPEC].set(m_dq[ARM_SPEC] / gear_ratio_arm)

    return jp.hstack([policy_q, policy_dq])
