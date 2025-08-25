import jax
import jax.numpy as jp

def act_fk()
'''Couple act from motor(low-level) space to joint(policy) space'''


def obs_ik()
'''Decouple obs from joint(policy) space to motor(low-level) space'''

# ref code
    def compute(
                self, 
                control_action: ArticulationActions, 
                joint_pos: torch.Tensor, 
                joint_vel: torch.Tensor) -> ArticulationActions:
        
        # knee command with gear ratio 

        # assume the command is in motor space (policy pov) , now transform it to joint space (sim pov)
        control_action.joint_positions[:,6] = control_action.joint_positions[:,6] / self.cfg.knee_gear_ratio
        control_action.joint_positions[:,7] = control_action.joint_positions[:,7] / self.cfg.knee_gear_ratio

        control_action.joint_velocities[:,6] = control_action.joint_velocities[:,6] / self.cfg.knee_gear_ratio
        control_action.joint_velocities[:,7] = control_action.joint_velocities[:,7] / self.cfg.knee_gear_ratio

        # knee-ankle coupling

        # get the knee joint positions
        q_j_knee_left = joint_pos[:,6]
        q_j_knee_right = joint_pos[:,7]
        qdot_j_knee_left = joint_vel[:,6]
        qdot_j_knee_right = joint_vel[:,7]

        # clip the ankle commands to the joint limits, q_m_ankle = nominal: 25.0 deg, min: -8.5 deg, max: 81.0 deg
        control_action.joint_positions[:,8] = torch.clip(control_action.joint_positions[:,8], min=-0.5846853, max= 0.977384)
        control_action.joint_positions[:,9] = torch.clip(control_action.joint_positions[:,9], min=-0.5846853, max= 0.977384)

        # assume the ankle command is in the motor space (coming from policy pov) , now transform it to joint space
        
        # for HECTOR_V1P5
        q_j_ankle_left = control_action.joint_positions[:,8] - q_j_knee_left 
        q_j_ankle_right = control_action.joint_positions[:,9] -q_j_knee_right
        qdot_j_ankle_left = control_action.joint_velocities[:,8]  - qdot_j_knee_left
        qdot_j_ankle_right = control_action.joint_velocities[:,9]  - qdot_j_knee_right
        
        # for HECTOR_V1P5_CI
        # q_j_ankle_left = control_action.joint_positions[:,8] - (q_j_knee_left + 1.5708)
        # q_j_ankle_right = control_action.joint_positions[:,9] -(q_j_knee_right + 1.5708)

        # overwrite the commands to the joint space (simulation pov)
        control_action.joint_positions[:,8] = q_j_ankle_left
        control_action.joint_positions[:,9] = q_j_ankle_right

        # dummy: store approximate torques for reward computation
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts

        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)

        return control_action