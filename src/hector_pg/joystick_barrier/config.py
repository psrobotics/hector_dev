from ml_collections import config_dict


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.5,
        history_len=1,
        soft_joint_pos_limit_factor=0.95,
        # OBS size
        obs_size=67,
        obs_hist_len=25,
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
                _reward_tracking_lin_vel=2.0,  # 2.0.
                _reward_tracking_ang_vel=1.5,  # 1.5
                # --- Base related rewards ---
                _cost_lin_vel_z=-1.0,
                _cost_ang_vel_xy=-0.5,
                _cost_orientation=-1.5,
                # --- Energy related rewards ---
                _cost_smoothness=-0.001,
                # --- Feet related rewards ---
                _reward_feet_height=2.0,
                _cost_feet_slip=-0.5,
                _cost_undesired_contact=-3.0,
                _cost_feet_dist=-0.0,
                # --- Other rewards ---
                _reward_alive=0.5,
                _cost_termination=-1.0,
                # --- Pose related rewards ---
                _cost_joint_pos_limits=-1.0,
                _cost_pose=-0.25,
            ),
            max_foot_height=0.08,
            max_contact_force=250.0,
            # Force threshold that holds as contact
            feet_f_contact=5.0,
            # Desired airtime within phase (1.0 scale)
            airtime=0.3,  # 0.45
            # In what precentage control will be ruleout
            default_p=0.1,
        ),
        push_config=config_dict.create(
            # Disable first to get a init policy
            enable=True,
            interval_range=[5.0, 10.0],  # [5.0, 10.0]
            magnitude_range=[0.1, 2.0],
        ),
        reset_config=config_dict.create(
            # [min, max] range for root x, y position noise
            root_pos_xy=[-0.5, 0.5],
            # [min, max] range for root yaw noise in radians
            root_yaw=[-3.14, 3.14],
            # [min, max] multiplicative scale for initial joint positions
            dof_pos_scale=[0.5, 1.5],
            # [min, max] range for root linear and angular velocity noise
            root_vel=[-0.5, 0.5],
            # [min, max] additive noise for general default pose
            dof_pos_add=[-0.05, 0.05],
            # [min, max] additive noise for specific joints (e.g., arms)
            dof_pos_add_special=[-0.1, 0.1],
            # [min, max] range for sampling the gait frequency
            gait_freq=[1.25, 1.5],
        ),
        # Resample within epsoide
        resample_step_interval=500,
        # Command sampling ranges
        lin_vel_x=[-1.0, 1.0],
        lin_vel_y=[-0.5, 0.5],
        ang_vel_yaw=[-1.0, 1.0],
        # Feet distance min  max
        f_dist_range=[0.08, 0.4],
        # Default body height
        body_height_default=0.55,
        impl="jax",  # "jax" or "warp"
        nconmax=8 * 8192,
        njmax=60,
    )
