"""
Environment implementation for ExoRL task with exoWE2_V2 robot.
The ExorlEnv is a custom DirectRLEnv designed for training the exoWE2_V2 exoskeleton robot to learn stable standing on flat terrain including various domain randomizations elements and disturbance forces.
"""
# flake8: noqa: W293

from __future__ import annotations
import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .exorl_env_cfg import ExorlEnvCfg


class ExorlEnv(DirectRLEnv):
    """Custom environment for the WE2 exoskeleton to learn stable standing on flat terrain."""

    cfg: ExorlEnvCfg

    # ------- Isaaclab defined functions -------

    def __init__(self, cfg: ExorlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        """Initialize environment, set up buffers and load motion data if imitation learning is enabled."""

        self.sim_dt = self.cfg.sim.dt
        self._processed_actions = torch.zeros(self.num_envs, 10, device=self.device)
        self._previous_processed_actions = torch.zeros(
            self.num_envs, 10, device=self.device
        )
        self._disturbance_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self._disturbance_cycle_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._disturbance_active_counter = self.cfg.disturbance_duration * torch.ones(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._previous_h = torch.zeros(self.num_envs, device=self.device)
        self._previous_h_torque = torch.zeros(self.num_envs, device=self.device)
        self._spawn_xy_positions = torch.zeros(self.num_envs, 2, device=self.device)
        
        # Actuator dynamics state buffers (if enabled)
        if self.cfg.apply_actuator_dynamics:
            # Convert state-space matrices from config lists to tensors
            self.sea_knee_H_A = torch.tensor(self.cfg.sea_knee_H_A, dtype=torch.float32, device=self.device)
            self.sea_knee_H_B = torch.tensor(self.cfg.sea_knee_H_B, dtype=torch.float32, device=self.device)
            self.sea_knee_H_C = torch.tensor(self.cfg.sea_knee_H_C, dtype=torch.float32, device=self.device)
            self.sea_knee_H_D = torch.tensor(self.cfg.sea_knee_H_D, dtype=torch.float32, device=self.device)
            
            self.sea_knee_Z_A = torch.tensor(self.cfg.sea_knee_Z_A, dtype=torch.float32, device=self.device)
            self.sea_knee_Z_B = torch.tensor(self.cfg.sea_knee_Z_B, dtype=torch.float32, device=self.device)
            self.sea_knee_Z_C = torch.tensor(self.cfg.sea_knee_Z_C, dtype=torch.float32, device=self.device)
            self.sea_knee_Z_D = torch.tensor(self.cfg.sea_knee_Z_D, dtype=torch.float32, device=self.device)
            
            self.sea_leg_H_A = torch.tensor(self.cfg.sea_leg_H_A, dtype=torch.float32, device=self.device)
            self.sea_leg_H_B = torch.tensor(self.cfg.sea_leg_H_B, dtype=torch.float32, device=self.device)
            self.sea_leg_H_C = torch.tensor(self.cfg.sea_leg_H_C, dtype=torch.float32, device=self.device)
            self.sea_leg_H_D = torch.tensor(self.cfg.sea_leg_H_D, dtype=torch.float32, device=self.device)
            
            self.sea_leg_Z_A = torch.tensor(self.cfg.sea_leg_Z_A, dtype=torch.float32, device=self.device)
            self.sea_leg_Z_B = torch.tensor(self.cfg.sea_leg_Z_B, dtype=torch.float32, device=self.device)
            self.sea_leg_Z_C = torch.tensor(self.cfg.sea_leg_Z_C, dtype=torch.float32, device=self.device)
            self.sea_leg_Z_D = torch.tensor(self.cfg.sea_leg_Z_D, dtype=torch.float32, device=self.device)
            
            # Initialize H and Z system states for all 8 joints
            # Shape: (num_envs, 8, 2) - 8 joints, 2 states each
            self._sea_x_H = torch.zeros(self.num_envs, 8, 2, dtype=torch.float32, device=self.device)
            self._sea_x_Z = torch.zeros(self.num_envs, 8, 2, dtype=torch.float32, device=self.device)
            
            # Create joint-to-model mapping (0-7 for actuated joints)
            # Knees (4-5) use Knee model, rest use Leg model
            self._sea_joint_to_model = torch.tensor(
                [1, 1, 1, 1, 0, 0, 1, 1],  # 0=knee, 1=leg
                device=self.device,
                dtype=torch.long
            )
            
            # Move state-space matrices to device and create model dict
            self._sea_models = {
                'knee': {
                    'H': {
                        'A': self.sea_knee_H_A,
                        'B': self.sea_knee_H_B,
                        'C': self.sea_knee_H_C,
                        'D': self.sea_knee_H_D,
                    },
                    'Z': {
                        'A': self.sea_knee_Z_A,
                        'B': self.sea_knee_Z_B,
                        'C': self.sea_knee_Z_C,
                        'D': self.sea_knee_Z_D,
                    }
                },
                'leg': {
                    'H': {
                        'A': self.sea_leg_H_A,
                        'B': self.sea_leg_H_B,
                        'C': self.sea_leg_H_C,
                        'D': self.sea_leg_H_D,
                    },
                    'Z': {
                        'A': self.sea_leg_Z_A,
                        'B': self.sea_leg_Z_B,
                        'C': self.sea_leg_Z_C,
                        'D': self.sea_leg_Z_D,
                    }
                }
            }
        
        # Load motion data for imitation learning if available
        self._motion_data_list = []  # List of all datasets: [fast, normal, slow]
        self._motion_data = None  # Currently selected dataset
        self._motion_data_length = 0
        self._selected_motion_dataset_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # Track which dataset each env is using
        self._current_motion_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Store XY offset between spawn position and motion data reference for pelvis reward
        self._pelvis_xy_offset = torch.zeros(self.num_envs, 2, device=self.device)
        
        # Motion logging for environment 0 (only when imitation learning is active)
        self._motion_log_enabled = self.cfg.imitation_learning_walking
        self._motion_log_target_q = []  # Target joint angles for env 0
        self._motion_log_target_dq = []  # Target joint velocities for env 0
        self._motion_log_target_pelvis = []  # Target pelvis position for env 0
        self._motion_log_target_pelvis_vel = []  # Target pelvis velocity for env 0
        self._motion_log_actual_q = []  # Actual joint angles for env 0
        self._motion_log_actual_dq = []  # Actual joint velocities for env 0
        self._motion_log_actual_pelvis = []  # Actual pelvis position for env 0
        self._motion_log_actual_pelvis_vel = []  # Actual pelvis velocity for env 0
        
        # CoM position logging for environment 0
        self._com_log_enabled = self.cfg.log_com_position
        self._com_log = []  # Center of mass positions [x, y, z] for env 0
        if self.cfg.imitation_learning_walking:
            try:
                import os
                # motion_files = ["motion_data_fast.npz", "motion_data_normal.npz", "motion_data_slow.npz"]
                # gait_names = ["fast", "normal", "slow"]
                
                # for debug purposes, just load the slow walking dataset
                motion_files = ["motion_data_slow.npz"]
                gait_names = ["slow"]
                
                for motion_file, gait_name in zip(motion_files, gait_names):
                    if os.path.exists(motion_file):
                        motion_npz = np.load(motion_file)
                        motion_data = {
                            'q_dq': torch.tensor(motion_npz['q_dq_data'], dtype=torch.float32, device=self.device),
                            'pelvis': torch.tensor(motion_npz['pelvis_data'], dtype=torch.float32, device=self.device),
                            'pelvis_vel_xy': torch.tensor(motion_npz['pelvis_vel_xy'], dtype=torch.float32, device=self.device),
                        }
                        self._motion_data_list.append(motion_data)
                        print(f"[INFO] Loaded {gait_name} motion data: {motion_data['q_dq'].shape[0]} samples")
                    else:
                        print(f"[WARN] Motion file '{motion_file}' not found.")
                
                if len(self._motion_data_list) == 0:
                    print(f"[WARN] No motion data files found. Imitation learning disabled.")
                    self.cfg.imitation_learning_walking = False
                else:
                    # Initialize with first dataset
                    self._motion_data = self._motion_data_list[0]
                    self._motion_data_length = self._motion_data['q_dq'].shape[0]
                    print(f"[INFO] Loaded {len(self._motion_data_list)} motion datasets for imitation learning with random gait selection")
                    print(f"[INFO] Motion tracking for env 0 enabled - will log target vs actual motion during training")
            except Exception as e:
                print(f"[WARN] Failed to load motion data: {e}. Imitation learning disabled.")
                self.cfg.imitation_learning_walking = False

        # Sub-rewards logging
        self.log_sub_rewards = self.cfg.log_sub_rewards
        if self.log_sub_rewards:
            self._sub_rewards_buffer = {
                "orientation_reward": [],
                "height_deviation_penalty": [],
                "torque_limit_violation_penalty": [],
                "action_change_penalty": [],
                "joint_angle_constraint_penalty": [],
                "cbf_h_value": [],
                "cbf_h_dot": [],
                "cbf_h_torque_value": [],
                "cbf_h_torque_dot": [],
                "joint_symmetry_penalty": [],
                "spawn_position_penalty": [],
                "walk_velocity_reward": [],
                "imitation_q_reward": [],
                "imitation_dq_reward": [],
                "imitation_pelvis_reward": [],
                "alive_reward": [],
                "early_termination_penalty": [],
                "total_reward": [],
            }
            self._step_counter = 0

        # Disturbance tests tracking
        self.disturbance_tests_mode = self.cfg.disturbance_tests_mode
        if self.disturbance_tests_mode:
            self._disturbance_tests_step_counter = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._disturbance_tests_survived = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            # Precompute direction vector in xy plane
            direction_rad = self.cfg.disturbance_tests_direction * np.pi / 180.0
            self._disturbance_tests_direction_x = np.cos(direction_rad)
            self._disturbance_tests_direction_y = np.sin(direction_rad)
            # Precompute force magnitudes for each environment (linearly spaced 0-100N)
            self._disturbance_tests_force_magnitudes = torch.linspace(
                0.0, self.cfg.disturbance_tests_max_force, self.num_envs, device=self.device
            )

        print(f"[DEBUG] Using sensor noise: {self.cfg.apply_sensor_noise}")
        print(
            f"[DEBUG] Using configuration randomization: {self.cfg.apply_configuration_randomization}"
        )
        print(f"[DEBUG] Using disturbance forces: {self.cfg.apply_disturbance_forces}")
        print(f"[DEBUG] Using torque control mode: {self.cfg.use_torque_control_mode}")
        print(f"[DEBUG] Using actuator dynamics: {self.cfg.apply_actuator_dynamics}")
        
        # Debug: Print reward scales
        print("\n[DEBUG] ========== REWARD SCALES ==========")
        print(f"[DEBUG] orientation_penalty_scale: {self.cfg.orientation_penalty_scale}")
        print(f"[DEBUG] height_reward_scale: {self.cfg.height_reward_scale}")
        print(f"[DEBUG] action_change_penalty_scale_position: {self.cfg.action_change_penalty_scale_position}")
        print(f"[DEBUG] action_change_penalty_scale_torque: {self.cfg.action_change_penalty_scale_torque}")
        print(f"[DEBUG] alive_reward: {self.cfg.alive_reward}")
        print(f"[DEBUG] early_termination_penalty_scale: {self.cfg.early_termination_penalty_scale}")
        print(f"[DEBUG] joint_symmetry_penalty_scale: {self.cfg.joint_symmetry_penalty_scale}")
        print(f"[DEBUG] cbf_angle_reward_scale: {self.cfg.cbf_angle_reward_scale}")
        print(f"[DEBUG] cbf_torque_reward_scale: {self.cfg.cbf_torque_reward_scale}")
        print(f"[DEBUG] walk_velocity_reward_scale: {self.cfg.walk_velocity_reward_scale}")
        print(f"[DEBUG] spawn_position_reward_scale: {self.cfg.spawn_position_reward_scale}")
        print(f"[DEBUG] imitation_q_reward_scale: {self.cfg.imitation_q_reward_scale}")
        print(f"[DEBUG] imitation_dq_reward_scale: {self.cfg.imitation_dq_reward_scale}")
        print(f"[DEBUG] imitation_pelvis_reward_scale: {self.cfg.imitation_pelvis_reward_scale}")
        print("[DEBUG] =====================================\n")

    def _setup_scene(self):
        """Set up the simulation scene with robot, ground, and lighting."""
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions, apply disturbance forces if enabled, and prepare for physics step."""
        if self.cfg.use_torque_control_mode:
            scaled_actions = self.cfg.action_scale_torque * actions
            scaled_actions = torch.clamp(scaled_actions, -self.cfg.joint_torque_saturation, self.cfg.joint_torque_saturation)
        else:
            scaled_actions = self.cfg.action_scale_position * actions

        self._processed_actions[:, :8] = scaled_actions
        self._processed_actions[:, 8:] = 0.0  # ankle adduction/abduction = passive

        if self.cfg.apply_disturbance_forces:
            self.apply_disturbance_forces()
        
        # Apply disturbance tests forces if enabled
        if self.disturbance_tests_mode:
            self.apply_disturbance_tests_forces()

    def _apply_action(self) -> None:
        """Apply processed actions to the robot in the simulation."""
        if self.cfg.use_torque_control_mode:
            # If actuator dynamics are enabled, compute actual torques after SEA model
            if self.cfg.apply_actuator_dynamics:
                # Get load-side velocities (simulated joint velocities for first 8 actuated joints)
                omega_l = self.robot.data.joint_vel[:, :8]
                
                # Compute actual torques through actuator dynamics
                tau_actual = self.compute_actuator_dynamics(self._processed_actions[:, :8], omega_l)
                tau_actual = torch.clamp(tau_actual, -self.cfg.joint_torque_saturation, self.cfg.joint_torque_saturation)
                
                # Create full action vector with computed torques
                actions_with_dynamics = self._processed_actions.clone()
                actions_with_dynamics[:, :8] = tau_actual
                
                self.robot.set_joint_effort_target(actions_with_dynamics)
            else:
                self.robot.set_joint_effort_target(self._processed_actions)
        else:
            self.robot.set_joint_position_target(self._processed_actions)

        self.robot.write_data_to_sim()

    def _get_observations(self) -> dict:
        """Construct observation from robot state with normalization."""
        joint_pos_act = self.robot.data.joint_pos[:, :8]
        joint_vel_act = self.robot.data.joint_vel[:, :8]
        base_quat = self.robot.data.root_quat_w
        base_ang_vel = self.robot.data.root_ang_vel_w

        # Convert quaternion to Euler angles and extract roll and pitch only
        euler_angles = self.quaternion_to_euler(base_quat)
        roll_pitch = euler_angles[:, :2]  # Only use roll and pitch, exclude yaw because it is not reliable in hardware

        if self.cfg.apply_sensor_noise:
            base_ang_vel, joint_pos_act, joint_vel_act, roll_pitch = self.add_noise_to_observations(base_ang_vel, joint_pos_act, joint_vel_act, roll_pitch)

        # Preprocess and normalize observations
        joint_pos_act_norm = self.preprocess_joint_pos_act(joint_pos_act)
        joint_vel_act_norm = self.preprocess_joint_vel_act(joint_vel_act)
        roll_pitch_norm = self.preprocess_roll_pitch(roll_pitch)
        base_ang_vel_norm = self.preprocess_base_ang_vel(base_ang_vel)
        
        # Get walk speed targets and normalize
        if self.cfg.imitation_learning_walking and self._motion_data_list:
            # Extract pelvis velocity from current motion data for each environment
            pelvis_vel_xy = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
            for i in range(self.num_envs):
                dataset_idx = self._selected_motion_dataset_idx[i].item()
                motion_step = self._current_motion_step[i].item()
                if dataset_idx < len(self._motion_data_list) and motion_step < self._motion_data_list[dataset_idx]['pelvis_vel_xy'].shape[0]:
                    pelvis_vel_xy[i] = self._motion_data_list[dataset_idx]['pelvis_vel_xy'][motion_step]
            
            speed_x_norm, speed_y_norm = self.preprocess_walk_speeds(pelvis_vel_xy[:, 0], pelvis_vel_xy[:, 1])
        else:
            # Use fixed target speeds from config
            speed_x = torch.full((self.num_envs,), self.cfg.target_walk_speed_x, device=self.device, dtype=torch.float32)
            speed_y = torch.full((self.num_envs,), self.cfg.target_walk_speed_y, device=self.device, dtype=torch.float32)
            speed_x_norm, speed_y_norm = self.preprocess_walk_speeds(speed_x, speed_y)
        
        speed_x_norm = speed_x_norm.unsqueeze(-1)
        speed_y_norm = speed_y_norm.unsqueeze(-1)

        # Concatenate normalized observations
        obs_list = [
            joint_pos_act_norm,
            joint_vel_act_norm,
            roll_pitch_norm,
            base_ang_vel_norm,
            speed_x_norm,
            speed_y_norm,
        ]
        
        # Check for any observation at the limit and print debug info
        for i, obs_component in enumerate(obs_list):
            if torch.any(obs_component[0] >= 1.0) or torch.any(obs_component[0] <= -1.0):
                print(f"Observation component {i} at limit: {obs_component[0].cpu().numpy()}")

        obs = torch.cat(obs_list, dim=-1)
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute reward for stable standing."""
        base_quat = self.robot.data.root_quat_w  # [w, x, y, z]
        base_pos = self.robot.data.root_pos_w
        base_lin_vel = self.robot.data.root_lin_vel_w  # root linear velocity in world frame
        joint_pos = self.robot.data.joint_pos  # shape: (num_envs, 10)
        if self.cfg.use_torque_control_mode:
            joint_effort = self._processed_actions[:, :8]  # shape: (num_envs, n_joints)
        else:
            joint_effort = self.robot.data.joint_effort_target[:, :8]

        # calculate and combine rewards
        orientation_reward = self.calculate_orientation_reward(base_quat)
        height_deviation_penalty = self.calculate_height_deviation_penalty(base_pos)
        torque_limit_violation_penalty, cbf_h_torque_value, cbf_h_torque_dot = self.calculate_torque_limit_violation_penalty(joint_effort)
        action_change_penalty = self.calculate_action_change_penalty()
        cbf_h_value, cbf_h_dot = self.calculate_joint_angle_constraints(joint_pos)
        joint_angle_constraint_penalty = self.calculate_constraint_penalty(cbf_h_value, cbf_h_dot)
        joint_symmetry_penalty = self.calculate_joint_symmetry_penalty(joint_pos)
        spawn_position_penalty = self.calculate_spawn_position_penalty(base_pos)
        walk_velocity_reward = self.calculate_walk_velocity_reward(base_lin_vel)
        alive_reward = self.calculate_alive_reward()
        early_termination_penalty = self.calculate_early_termination_penalty(base_pos)
        imitation_q_reward = torch.zeros(self.num_envs, device=self.device)
        imitation_dq_reward = torch.zeros(self.num_envs, device=self.device)
        imitation_pelvis_reward = torch.zeros(self.num_envs, device=self.device)
        imitation_q_reward, imitation_dq_reward, imitation_pelvis_reward = self.calculate_imitation_rewards()

        total_reward = (
            orientation_reward
            + height_deviation_penalty
            + torque_limit_violation_penalty
            + action_change_penalty
            + joint_angle_constraint_penalty
            + joint_symmetry_penalty
            + spawn_position_penalty
            + walk_velocity_reward
            + imitation_q_reward
            + imitation_dq_reward
            + imitation_pelvis_reward
            + alive_reward
            + early_termination_penalty
        )

        # Log sub-rewards if enabled
        if self.log_sub_rewards:
            self._store_reward_metrics(
                orientation_reward,
                height_deviation_penalty,
                torque_limit_violation_penalty,
                action_change_penalty,
                joint_angle_constraint_penalty,
                cbf_h_value,
                cbf_h_dot,
                cbf_h_torque_value,
                cbf_h_torque_dot,
                joint_symmetry_penalty,
                spawn_position_penalty,
                walk_velocity_reward,
                imitation_q_reward,
                imitation_dq_reward,
                imitation_pelvis_reward,
                alive_reward,
                early_termination_penalty,
                total_reward
            )
        
        # Log center of mass position if enabled
        if self._com_log_enabled:
            self._com_log.append(base_pos[0, :3].cpu().numpy())

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions.

        Termination conditions:
        - Fallen: base height drops below threshold (cannot recover)
        - Time limit: episode_length_s exceeded
        """
        base_pos = self.robot.data.root_pos_w

        # Check if fallen (base too low)
        fallen = base_pos[:, 2] < self.cfg.fall_height_threshold

        # Track survival in disturbance tests mode, the variable _disturbance_tests_survived will be True for environments that have not fallen at any point during the episode, and False for environments that have fallen at least once. This allows us to measure how many environments survive the entire disturbance test without falling.
        if self.disturbance_tests_mode:
            self._disturbance_tests_survived = self._disturbance_tests_survived & ~fallen

        # Debug: Print when env 0 is considered fallen
        # if fallen[0]:
        #     base_height_env0 = base_pos[0, 2].item()
        #     print(f"[DEBUG] Env 0: FALLEN - base height {base_height_env0:.2f}m below threshold {self.cfg.fall_height_threshold}m")

        # Check time limit
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return fallen, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments to initial state with randomized joint positions."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        num_reset_envs = len(env_ids)

        if self.cfg.apply_configuration_randomization:
            # Randomize joint positions within configured limits
            random_joint_pos = self.randomize_joint_positions(env_ids, num_reset_envs)
        else:
            # Use default joint positions from robot data
            random_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()

        # Set all joint velocities to zero for stable initial conditions
        joint_vel = torch.zeros_like(random_joint_pos)
        self.robot.write_joint_state_to_sim(random_joint_pos, joint_vel, None, env_ids)

        # Reset root pose and velocity to default
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        # ensure proper initial height, for some reason the exoskeleton would otherwise spawn in the ground
        root_state[:, 2] += 0.98
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        # Reset processed actions
        self._processed_actions[env_ids] = 0.0
        self._previous_processed_actions[env_ids] = 0.0

        # Reset control barrier function state
        self._previous_h[env_ids] = 0.0
        self._previous_h_torque[env_ids] = 0.0

        # Store spawn XY position for spawn position penalty
        self._spawn_xy_positions[env_ids] = self.robot.data.root_pos_w[env_ids, :2]

        # Reset disturbance forces and timers
        self._disturbance_forces[env_ids] = 0.0
        self._disturbance_cycle_counter[env_ids] = 0
        self._disturbance_active_counter[env_ids] = self.cfg.disturbance_duration
        if hasattr(self, "_disturbance_offset"):
            self._disturbance_offset[env_ids] = 0.0

        # Reset actuator dynamics state buffers
        if self.cfg.apply_actuator_dynamics:
            self._sea_x_H[env_ids] = 0.0
            self._sea_x_Z[env_ids] = 0.0

        # Reset motion data step counter for imitation learning
        self._current_motion_step[env_ids] = 0
        
        # Randomly select a motion dataset for each environment (with equal probability)
        if self.cfg.imitation_learning_walking and self._motion_data_list:
            num_datasets = len(self._motion_data_list)
            random_dataset_indices = torch.randint(0, num_datasets, (num_reset_envs,), device=self.device)
            self._selected_motion_dataset_idx[env_ids] = random_dataset_indices
        
        # Store pelvis XY offset for imitation learning reward (accounts for environment spawn position and selected dataset)
        if self.cfg.imitation_learning_walking and self._motion_data_list:
            for i, env_id in enumerate(env_ids):
                dataset_idx = self._selected_motion_dataset_idx[env_id].item()
                if dataset_idx < len(self._motion_data_list):
                    dataset = self._motion_data_list[dataset_idx]
                    self._pelvis_xy_offset[env_id] = root_state[i, :2] - dataset['pelvis'][0, :2]

        # Call parent reset to reset episode buffers
        super()._reset_idx(env_ids)

    # ---------- Helper functions ----------
    def _store_reward_metrics(self, orientation_reward, height_deviation_penalty, torque_limit_violation_penalty, action_change_penalty, joint_angle_constraint_penalty, cbf_h_value, cbf_h_dot, cbf_h_torque_value, cbf_h_torque_dot, joint_symmetry_penalty, spawn_position_penalty, walk_velocity_reward, imitation_q_reward, imitation_dq_reward, imitation_pelvis_reward, alive_reward, early_termination_penalty, total_reward):
        """Log sub-rewards to internal buffer for later saving."""
        LOG_FIRST_N_ENVS = 50

        self._sub_rewards_buffer["orientation_reward"].append(orientation_reward[:LOG_FIRST_N_ENVS].cpu().numpy())
        self._sub_rewards_buffer["height_deviation_penalty"].append(height_deviation_penalty[:LOG_FIRST_N_ENVS].cpu().numpy())
        self._sub_rewards_buffer["torque_limit_violation_penalty"].append(torque_limit_violation_penalty[:LOG_FIRST_N_ENVS].cpu().numpy())
        self._sub_rewards_buffer["action_change_penalty"].append(action_change_penalty[:LOG_FIRST_N_ENVS].cpu().numpy())
        self._sub_rewards_buffer["joint_angle_constraint_penalty"].append(joint_angle_constraint_penalty[:LOG_FIRST_N_ENVS].cpu().numpy())
        self._sub_rewards_buffer["cbf_h_value"].append(cbf_h_value[:LOG_FIRST_N_ENVS].cpu().numpy())
        self._sub_rewards_buffer["cbf_h_dot"].append(cbf_h_dot[:LOG_FIRST_N_ENVS].cpu().numpy())
        self._sub_rewards_buffer["cbf_h_torque_value"].append(cbf_h_torque_value[:LOG_FIRST_N_ENVS].cpu().numpy())
        self._sub_rewards_buffer["cbf_h_torque_dot"].append(cbf_h_torque_dot[:LOG_FIRST_N_ENVS].cpu().numpy())
        self._sub_rewards_buffer["joint_symmetry_penalty"].append(joint_symmetry_penalty[:LOG_FIRST_N_ENVS].cpu().numpy())
        self._sub_rewards_buffer["spawn_position_penalty"].append(spawn_position_penalty[:LOG_FIRST_N_ENVS].cpu().numpy())
        self._sub_rewards_buffer["walk_velocity_reward"].append(walk_velocity_reward[:LOG_FIRST_N_ENVS].cpu().numpy())
        self._sub_rewards_buffer["imitation_q_reward"].append(imitation_q_reward[:LOG_FIRST_N_ENVS].cpu().numpy())
        self._sub_rewards_buffer["imitation_dq_reward"].append(imitation_dq_reward[:LOG_FIRST_N_ENVS].cpu().numpy())
        self._sub_rewards_buffer["imitation_pelvis_reward"].append(imitation_pelvis_reward[:LOG_FIRST_N_ENVS].cpu().numpy())
        self._sub_rewards_buffer["alive_reward"].append(alive_reward[:LOG_FIRST_N_ENVS].cpu().numpy())
        self._sub_rewards_buffer["early_termination_penalty"].append(early_termination_penalty[:LOG_FIRST_N_ENVS].cpu().numpy())
        self._sub_rewards_buffer["total_reward"].append(total_reward[:LOG_FIRST_N_ENVS].cpu().numpy())
        self._step_counter += 1

    def randomize_joint_positions(self, env_ids, num_reset_envs):
        joint_angle_min_rad = torch.tensor(self.cfg.joint_angle_min_rad, device=self.device, dtype=torch.float32)
        joint_angle_max_rad = torch.tensor(self.cfg.joint_angle_max_rad, device=self.device, dtype=torch.float32)
        std_rad = torch.tensor(self.cfg.joint_angle_config_randomization_std_rad, device=self.device, dtype=torch.float32)

        # Get default joint positions for these environments
        default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()

        random_joint_pos = torch.zeros(num_reset_envs, 10, device=self.device, dtype=torch.float32)
        for i in range(8):
            # Randomize around default configuration with normal distribution using per-joint std
            perturbation = torch.randn(num_reset_envs, device=self.device, dtype=torch.float32) * std_rad[i]
            random_joint_pos[:, i] = default_joint_pos[:, i] + perturbation
            # Clip to min/max bounds
            random_joint_pos[:, i] = torch.clamp(random_joint_pos[:, i], joint_angle_min_rad[i], joint_angle_max_rad[i])

        random_joint_pos[:, 8:] = default_joint_pos[:, 8:]  # passive joints

        return random_joint_pos

    def save_sub_rewards(self, log_dir: str) -> None:
        """Save logged sub-rewards to a compressed NPZ file.

        Args:
            log_dir: Directory to save the sub-rewards log file.
        """
        if not self.log_sub_rewards or self._step_counter == 0:
            return

        import os
        from datetime import datetime

        os.makedirs(log_dir, exist_ok=True)

        # Convert lists to numpy arrays
        sub_rewards_data = {}
        for key, values in self._sub_rewards_buffer.items():
            if values:
                # Stack all values along time axis
                sub_rewards_data[key] = np.array(values)

        # Save to compressed NPZ file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        log_file = os.path.join(log_dir, f"{timestamp}_sub_rewards.npz")
        np.savez_compressed(log_file, **sub_rewards_data)

        print(f"[INFO] Sub-rewards logged to: {log_file}")
        print(
            f"[INFO] Logged {self._step_counter} steps with sub-reward data for {len(sub_rewards_data)} reward components."
        )

    def save_motion_logs(self, log_dir: str) -> None:
        """Save logged motion data for environment 9 to a compressed NPZ file.
        
        Saves target vs actual motion including joint angles, velocities, and pelvis position/velocity.

        Args:
            log_dir: Directory to save the motion log file.
        """
        if not self._motion_log_enabled or len(self._motion_log_target_q) == 0:
            print("[INFO] No motion data to log (imitation learning disabled or no data recorded)")
            return

        import os
        from datetime import datetime

        os.makedirs(log_dir, exist_ok=True)

        # Convert lists to numpy arrays
        motion_data = {
            'target_joint_angles': np.array(self._motion_log_target_q),
            'target_joint_velocities': np.array(self._motion_log_target_dq),
            'target_pelvis_position': np.array(self._motion_log_target_pelvis),
            'target_pelvis_velocity': np.array(self._motion_log_target_pelvis_vel),
            'actual_joint_angles': np.array(self._motion_log_actual_q),
            'actual_joint_velocities': np.array(self._motion_log_actual_dq),
            'actual_pelvis_position': np.array(self._motion_log_actual_pelvis),
            'actual_pelvis_velocity': np.array(self._motion_log_actual_pelvis_vel),
        }

        # Save to compressed NPZ file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        log_file = os.path.join(log_dir, f"{timestamp}_env0_motion_log.npz")
        np.savez_compressed(log_file, **motion_data)

        num_steps = len(self._motion_log_target_q)
        print(f"[INFO] Motion logs for environment 0 saved to: {log_file}")
        print(f"[INFO] Recorded {num_steps} steps of target vs actual motion data")
        print(f"[INFO] - Target/Actual joint angles: {motion_data['target_joint_angles'].shape}")
        print(f"[INFO] - Target/Actual joint velocities: {motion_data['target_joint_velocities'].shape}")
        print(f"[INFO] - Target/Actual pelvis position: {motion_data['target_pelvis_position'].shape}")
        print(f"[INFO] - Target/Actual pelvis velocity: {motion_data['target_pelvis_velocity'].shape}")

    def save_com_position(self, log_dir: str) -> None:
        """Save logged center of mass positions to a CSV file.
        
        Saves x, y, z coordinates of environment 0's center of mass.

        Args:
            log_dir: Directory to save the CSV file.
        """
        if not self._com_log_enabled or len(self._com_log) == 0:
            print("[INFO] No CoM data to log (CoM logging disabled or no data recorded)")
            return

        import os
        import csv
        from datetime import datetime

        os.makedirs(log_dir, exist_ok=True)

        # Convert list to numpy array
        com_data = np.array(self._com_log)  # shape: (num_steps, 3)

        # Save to CSV file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        log_file = os.path.join(log_dir, f"{timestamp}_env0_com_position.csv")
        
        with open(log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x', 'y', 'z'])
            writer.writerows(com_data)

        num_steps = len(self._com_log)
        print(f"[INFO] CoM positions for environment 0 saved to: {log_file}")
        print(f"[INFO] Recorded {num_steps} timesteps of center of mass data")

    @staticmethod
    def quaternion_to_euler(quat_w: torch.Tensor) -> torch.Tensor:
        """Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw].

        Uses ZYX (yaw-pitch-roll) convention.

        Args:
            quat_w: Quaternion tensor of shape (*, 4) with components [w, x, y, z]

        Returns:
            Euler angles tensor of shape (*, 3) with [roll, pitch, yaw] in radians
        """
        w = quat_w[..., 0]
        x = quat_w[..., 1]
        y = quat_w[..., 2]
        z = quat_w[..., 3]

        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        return torch.stack([roll, pitch, yaw], dim=-1)

    def calculate_joint_symmetry_penalty(self, joint_pos):
        """Calculate penalty for asymmetry between left and right joint pairs.
        
        Penalizes when left and right joints are far apart for:
        - Hip Abduction/Adduction (indices 0-1)
        - Hip Flexion/Extension (indices 2-3)
        - Knee Flexion/Extension (indices 4-5)
        - Ankle Flexion/Extension (indices 6-7)
        
        Uses MSE (mean squared error) metric to penalize asymmetry.
        """
        joint_pos_actuated = joint_pos[:, :8]
        
        # Compute pairwise differences between left and right joints
        # Shape: (num_envs, 4) for the 4 joint pair types
        asymmetries = torch.abs(
            joint_pos_actuated[:, ::2] - joint_pos_actuated[:, 1::2]
        )  # indices: [0-1, 2-3, 4-5, 6-7]
        
        # MSE across all joint pairs
        joint_symmetry_penalty = -torch.sum(asymmetries**2, dim=1) * self.cfg.joint_symmetry_penalty_scale
        
        return joint_symmetry_penalty

    def calculate_constraint_penalty(self, cbf_h_value, cbf_h_dot):
        return cbf_h_dot + cbf_h_value

    def calculate_joint_angle_constraints(self, joint_pos):
        """Calculate control barrier function (CBF) components for joint angle constraints.

        The barrier function penalizes when joints approach their limits and couples
        position and velocity information through finite-difference approximation.

        Inspired from paper: https://doi.org/10.48550/arXiv.2403.01410
        """
        joint_angle_min = torch.tensor(
            self.cfg.joint_angle_min_constraint, device=self.device, dtype=torch.float32
        )
        joint_angle_max = torch.tensor(
            self.cfg.joint_angle_max_constraint, device=self.device, dtype=torch.float32
        )
        joint_pos_actuated = joint_pos[:, :8]

        # Compute barrier function
        exp_upper = torch.exp(self.cfg.cbf_angle_delta_b * (joint_pos_actuated - joint_angle_max))
        exp_lower = torch.exp(self.cfg.cbf_angle_delta_b * (joint_angle_min - joint_pos_actuated))
        h_current = torch.sum(
            self.cfg.cbf_angle_delta_a * (1.0 - (exp_upper + exp_lower)),
            dim=1
        )  # shape: (num_envs,)
        h_current = torch.clamp(h_current, max=0.0)  # Ensure that within safe region (h >= 0) the value is constant to avoid penalizing unnecessarily (aka to avoid converging towards the center of the joint range instead of just staying within limits)

        # Finite difference approximation: ḣ = (h_t - h_{t-1}) / Δt
        h_dot = torch.clamp((h_current - self._previous_h) / self.sim_dt * self.cfg.cbf_angle_beta, max=10.0)  # shape: (num_envs,)

        # Static penalty component: γ·h(s)
        h_gamma = self.cfg.cbf_angle_gamma * h_current

        # Update stored h for next timestep
        self._previous_h = h_current.clone()

        # scale with reward scale
        h_gamma *= self.cfg.cbf_angle_reward_scale
        h_dot *= self.cfg.cbf_angle_reward_scale

        return h_gamma, h_dot

    def calculate_action_change_penalty(self):
        """Calculate penalty for changes in actions between time steps.
        
        Only applies penalty when no disturbance forces are active.
        """
        action_change = self._processed_actions[:, :8] - self._previous_processed_actions[:, :8]
        if self.cfg.use_torque_control_mode:
            action_change_penalty = -torch.sum(action_change**2, dim=1) * self.cfg.action_change_penalty_scale_torque
        else:
            action_change_penalty = -torch.sum(action_change**2, dim=1) * self.cfg.action_change_penalty_scale_position
        self._previous_processed_actions = self._processed_actions.clone()
        
        # Only apply penalty when no disturbance forces are currently active
        # Counter >= disturbance_duration means no active disturbance, because then the disturbance has passed already
        no_disturbance_mask = self._disturbance_active_counter >= self.cfg.disturbance_duration
        action_change_penalty = action_change_penalty * no_disturbance_mask.float()
        
        # Debug: Print when env 0's action change penalty is turned off due to active disturbance
        if not no_disturbance_mask[0]:
            print(
                f"[DEBUG] Env 0: Action change penalty OFF - disturbance active (counter: {self._disturbance_active_counter[0].item()} / {self.cfg.disturbance_duration})"
            )
        
        return action_change_penalty

    def calculate_torque_limit_violation_penalty(self, joint_effort):
        """Calculate penalty for joint torques exceeding limits using quadratic Control Barrier Function.
        
        Uses quadratic barrier function: h = sum(delta_a * (tau_max^2 - tau_l^2))
        Only applies in torque control mode.
        
        Returns:
            Tuple of (penalty, h_torque_current, h_torque_dot)
        
        Inspired from paper: https://doi.org/10.48550/arXiv.2403.01410
        """
        # if not self.cfg.use_torque_control_mode:
        #     # No penalty when not in torque control mode
        #     print("[DEBUG] Torque limit violation penalty is zero because not in torque control mode. If you see this message you should implement that function.")
        #     return torch.zeros(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device)

        # Get only the actuated joints (first 8)
        actuated_torques = joint_effort[:, :8]  # shape: (num_envs, 8)

        # Compute quadratic barrier function: h = sum(delta_a * (tau_max^2 - tau^2))
        # Since tau_min = -tau_max we rearranged the formula to this form
        # h > 0 when |tau| < tau_max (safe region)
        # h = 0 when |tau| = tau_max (boundary)
        # h < 0 when |tau| > tau_max (violation)
        tau_max = self.cfg.joint_torque_saturation
        h_torque_current = torch.sum(
            self.cfg.cbf_torque_delta_a * (tau_max**2 - actuated_torques**2),
            dim=1
        )  # shape: (num_envs,)

        # Finite difference approximation: ḣ = (h_t - h_{t-1}) / Δt
        h_torque_dot = torch.clamp((h_torque_current - self._previous_h_torque) / self.sim_dt, max=1.0)  # shape: (num_envs,)
        self._previous_h_torque = h_torque_current.clone()
        
        # (γ·h(s) + ḣ) * reward_scale
        penalty = (self.cfg.cbf_torque_gamma * h_torque_current + self.cfg.cbf_torque_beta * h_torque_dot) * self.cfg.cbf_torque_reward_scale
        
        # Debug: Print when env 0 has violating joint torques
        # if h_torque_current[0] < 0.0:
        #     print(
        #         f"[DEBUG] Env 0: Torque violation - h_torque: {h_torque_current[0].item():.2f}, torque: {actuated_torques[0].cpu().numpy()}, tau_max: {tau_max}"
        #     )

        return penalty, h_torque_current, h_torque_dot

    def calculate_height_deviation_penalty(self, base_pos):
        """Calculate height deviation penalty based on deviation from target height."""
        height_deviation_penalty = (
            -torch.square(base_pos[:, 2] - self.cfg.target_height)
            * self.cfg.height_reward_scale
        )
        return height_deviation_penalty

    def calculate_orientation_reward(self, base_quat):
        """Calculate orientation reward based on deviation from upright orientation."""
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=base_quat.dtype)
        quat_error = torch.sum((base_quat - identity_quat) ** 2, dim=1)
        orientation_reward = -quat_error * self.cfg.orientation_penalty_scale
        return orientation_reward

    def calculate_early_termination_penalty(self, base_pos):
        """Calculate early termination penalty for fallen robots that have not reached time limit."""
        fallen = base_pos[:, 2] < self.cfg.fall_height_threshold
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        early_termination_penalty = torch.zeros(self.num_envs, device=self.device)
        early_termination_penalty[fallen & ~time_out] = -self.cfg.early_termination_penalty_scale
        return early_termination_penalty

    def calculate_alive_reward(self):
        """Calculate alive reward for robots that are still active."""
        alive_reward = torch.ones(self.num_envs, device=self.device) * self.cfg.alive_reward
        return alive_reward

    def calculate_spawn_position_penalty(self, base_pos):
        """Calculate penalty for deviating from spawn position in x-y plane.
        """
        pos_xy = base_pos[:, :2]  # (num_envs, 2)
        pos_error = pos_xy - self._spawn_xy_positions  # (num_envs, 2)
        spawn_position_penalty = -torch.sum(pos_error**2, dim=1) * self.cfg.spawn_position_reward_scale
        
        # print debug info for environment 0 about spawn position penalty
        if False:
            print(
                f"[DEBUG] Env 0: Spawn position penalty: {spawn_position_penalty[0].item():.2f}, pos_xy: {pos_xy[0].cpu().numpy()}, spawn_xy: {self._spawn_xy_positions[0].cpu().numpy()}, error: {pos_error[0].cpu().numpy()}"
            )

        return spawn_position_penalty

    def calculate_walk_velocity_reward(self, base_lin_vel):
        """Calculate reward for matching target walking velocity in x-y plane.
        """
        vel_x = base_lin_vel[:, 0]
        vel_y = base_lin_vel[:, 1]

        vx_error = vel_x - self.cfg.target_walk_speed_x
        vy_error = vel_y - self.cfg.target_walk_speed_y

        walk_velocity_reward = -(vx_error**2 + vy_error**2) * self.cfg.walk_velocity_reward_scale

        return walk_velocity_reward

    def calculate_imitation_rewards(self):
        """Calculate rewards for imitating motion from dataset.
        Supports multiple datasets (fast, normal, slow) with per-environment selection.
        Logs motion data for environment 0 during training.
        """
        if not self._motion_data_list or len(self._motion_data_list) == 0:
            return torch.zeros(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device)

        # Get current joint positions and velocities (first 8 actuated joints)
        current_q = self.robot.data.joint_pos[:, :8]  # (num_envs, 8)
        current_dq = self.robot.data.joint_vel[:, :8]  # (num_envs, 8)
        current_pelvis = self.robot.data.root_pos_w[:, :3]  # (num_envs, 3)
        current_pelvis_vel = self.robot.data.root_lin_vel_w[:, :2]  # (num_envs, 2) - XY velocity only

        # Initialize rewards
        q_reward = torch.zeros(self.num_envs, device=self.device)
        dq_reward = torch.zeros(self.num_envs, device=self.device)
        pelvis_reward = torch.zeros(self.num_envs, device=self.device)

        # Process each environment with its selected dataset
        for env_idx in range(self.num_envs):
            dataset_idx = self._selected_motion_dataset_idx[env_idx].item()
            motion_step = self._current_motion_step[env_idx].item()
            
            # Get the selected dataset
            if dataset_idx >= len(self._motion_data_list):
                continue
            
            dataset = self._motion_data_list[dataset_idx]
            dataset_length = dataset['q_dq'].shape[0]
            
            # Clamp motion step to dataset bounds
            clamped_step = min(motion_step, dataset_length - 1)
            
            # Extract target states from the current dataset
            target_q = dataset['q_dq'][clamped_step, :8]  # (8,)
            target_dq = dataset['q_dq'][clamped_step, 8:16]  # (8,)
            target_pelvis = dataset['pelvis'][clamped_step, :3]  # (3,)
            target_pelvis_vel = dataset['pelvis_vel_xy'][clamped_step, :2]  # (2,)
            
            # Compute errors for this environment
            q_error = torch.sum((current_q[env_idx] - target_q)**2)  # scalar
            dq_error = torch.sum((current_dq[env_idx] - target_dq)**2)  # scalar
            
            # Adjust current pelvis XY to account for spawn offset, keeping Z as-is
            current_pelvis_adjusted = current_pelvis[env_idx].clone()
            current_pelvis_adjusted[:2] = current_pelvis[env_idx, :2] - self._pelvis_xy_offset[env_idx]
            # since we spawn the exoskeleton at a height of 0.9m, we need to add that as an offset to the motion data
            current_pelvis_adjusted[2] = current_pelvis[env_idx, 2] - 0.9
            pelvis_error = torch.sum((current_pelvis_adjusted - target_pelvis)**2)  # scalar
            
            # Compute rewards (negative squared error, scaled)
            q_reward[env_idx] = -q_error * self.cfg.imitation_q_reward_scale
            dq_reward[env_idx] = -dq_error * self.cfg.imitation_dq_reward_scale
            pelvis_reward[env_idx] = -pelvis_error * self.cfg.imitation_pelvis_reward_scale
            
            # Log motion data for environment 9
        if self._motion_log_enabled and env_idx == 9:
                self._motion_log_target_q.append(target_q.cpu().numpy())
                self._motion_log_target_dq.append(target_dq.cpu().numpy())
                self._motion_log_target_pelvis.append(target_pelvis.cpu().numpy())
                self._motion_log_target_pelvis_vel.append(target_pelvis_vel.cpu().numpy())
                self._motion_log_actual_q.append(current_q[env_idx].cpu().numpy())
                self._motion_log_actual_dq.append(current_dq[env_idx].cpu().numpy())
                self._motion_log_actual_pelvis.append(current_pelvis_adjusted.cpu().numpy())
                self._motion_log_actual_pelvis_vel.append(current_pelvis_vel[env_idx].cpu().numpy())

        # Increment motion data step counter for next step
        self._current_motion_step += 1

        return q_reward, dq_reward, pelvis_reward

    def apply_disturbance_forces(self) -> None:
        """Apply timer-based disturbance forces with smooth ramps to the robot base for robustness training.

        Forces are applied when cycle counter reaches disturbance_cycle_interval,
        then continuously for disturbance_duration cycles with smooth ramp-in and ramp-out.
        Force direction and application point are randomized for each disturbance event.
        Smooth ramping prevents step-input shocks and allows the policy to adjust gradually.
        """
        # Increment cycle counter for all environments
        self._disturbance_cycle_counter += 1

        # Check which environments should start a new disturbance
        start_disturbance = (
            self._disturbance_cycle_counter >= self.cfg.disturbance_cycle_interval
        )

        # For environments starting disturbance, reset counter and initialize active counter
        self._disturbance_cycle_counter[start_disturbance] = 0
        self._disturbance_active_counter[start_disturbance] = 0

        # Check which environments are currently in active disturbance phase
        is_active = self._disturbance_active_counter < self.cfg.disturbance_duration

        if not is_active.any():
            # No active disturbances, disable external wrench
            self._disturbance_forces.zero_()
            self.robot.set_external_force_and_torque(
                forces=torch.zeros(self.num_envs, 1, 3, device=self.device),
                torques=torch.zeros(self.num_envs, 1, 3, device=self.device),
                body_ids=[0],
                is_global=True,
            )
            return

        # For new disturbances, sample random force direction and magnitude
        newly_active = start_disturbance & (self._disturbance_active_counter == 0)

        # Sample force magnitude
        force_magnitude = (
            torch.rand(self.num_envs, device=self.device)
            * (self.cfg.disturbance_force_max - self.cfg.disturbance_force_min)
            + self.cfg.disturbance_force_min
        )

        # Sample random force direction (uniformly in x-y plane, no z component)
        force_direction = torch.zeros(self.num_envs, 3, device=self.device)
        force_direction[:, :2] = torch.randn(self.num_envs, 2, device=self.device)
        force_direction[:, :2] = force_direction[:, :2] / (
            torch.norm(force_direction[:, :2], dim=1, keepdim=True) + 1e-6
        )

        # Combine magnitude and direction (this is the FULL force, before ramping)
        full_force = force_direction * force_magnitude.unsqueeze(-1)
        
        # Store full forces for newly active environments
        if not hasattr(self, "_disturbance_forces_full"):
            self._disturbance_forces_full = torch.zeros(self.num_envs, 3, device=self.device)
        self._disturbance_forces_full[newly_active] = full_force[newly_active]

        # ===== COMPUTE RAMP FACTOR FOR SMOOTH FORCE APPLICATION =====
        ramp_factor = torch.ones(self.num_envs, device=self.device)
        
        # During ramp-in phase (smooth acceleration of force: 0 → 1)
        ramp_in_phase = (self._disturbance_active_counter < self.cfg.disturbance_ramp_in_steps)
        if ramp_in_phase.any():
            ramp_up = (self._disturbance_active_counter[ramp_in_phase] + 1).float() / self.cfg.disturbance_ramp_in_steps
            ramp_factor[ramp_in_phase] = torch.clamp(ramp_up, max=1.0)
        
        # During ramp-out phase (smooth deceleration of force: 1 → 0)
        ramp_out_phase = (
            self._disturbance_active_counter >= (self.cfg.disturbance_duration - self.cfg.disturbance_ramp_out_steps)
        )
        if ramp_out_phase.any():
            steps_in_ramp_out = (
                self.cfg.disturbance_duration - self._disturbance_active_counter[ramp_out_phase]
            ).float()
            ramp_down = steps_in_ramp_out / self.cfg.disturbance_ramp_out_steps
            ramp_factor[ramp_out_phase] = torch.clamp(ramp_down, min=0.0, max=1.0)
        
        # Apply ramp factor to the stored full forces
        self._disturbance_forces = self._disturbance_forces_full * ramp_factor.unsqueeze(-1)
        
        # Only apply force to actively disturbed environments
        self._disturbance_forces[~is_active] = 0.0

        # Get application point (mostly at CoM but with some randomization)
        if self.cfg.disturbance_com_offset_range > 0:
            # Random offset from CoM - resample for each newly started disturbance (no x offset, y-z only)
            offset = torch.zeros(self.num_envs, 3, device=self.device)
            offset[:, 1:3] = torch.rand(self.num_envs, 2, device=self.device) * 2 - 1  # [-1, 1]
            offset = offset * self.cfg.disturbance_com_offset_range
            # Store offset for environments that just started (for consistency during active phase)
            if not hasattr(self, "_disturbance_offset"):
                self._disturbance_offset = offset.clone()
            else:
                self._disturbance_offset[newly_active] = offset[newly_active]
            offset = self._disturbance_offset
            # Only apply offset to actively disturbed environments
            offset[~is_active] = 0.0
        else:
            offset = torch.zeros(self.num_envs, 3, device=self.device)

        # Debug: Print when env 0 is actively experiencing disturbance
        if is_active[0]:
            ramp_factor_env0 = ramp_factor[0].item()
            force_magnitude_env0 = force_magnitude[0].item()
            disturbance_force_env0 = self._disturbance_forces[0].cpu().numpy()
            active_step = self._disturbance_active_counter[0].item()
            print(f"[DEBUG] Env 0: Disturbance ACTIVE - step {active_step}/{self.cfg.disturbance_duration}, "
                  f"magnitude={force_magnitude_env0:.2f}N, ramp_factor={ramp_factor_env0:.3f}, "
                  f"ramped_force={disturbance_force_env0}, offset={offset[0].cpu().numpy()}")

        # Apply the force to the robot base
        forces = self._disturbance_forces.unsqueeze(1)  # (num_envs, 1, 3)
        torques = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self.robot.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            positions=offset.unsqueeze(1),  # (num_envs, 1, 3)
            body_ids=[0],  # Apply to root body
            is_global=True,
        )
        self.robot.write_data_to_sim()

        # Increment active counter for environments that are currently active
        self._disturbance_active_counter[is_active] += 1

    def apply_disturbance_tests_forces(self) -> None:
        """Apply disturbance forces for the disturbance tests mode.
        
        In disturbance tests mode:
        - After 3 seconds (300 steps at 100 Hz), apply a force for 100ms (10 steps)
        - Force magnitude varies linearly from 0N (env 0) to 100N (env 9999)
        - Force is applied in xy plane at the direction specified by disturbance_tests_direction
        """
        # Increment step counter
        self._disturbance_tests_step_counter += 1
        
        # Check if we should apply force (after delay and within duration window)
        delay_steps = self.cfg.disturbance_tests_delay_steps
        duration_steps = self.cfg.disturbance_tests_force_duration_steps
        should_apply_force = (self._disturbance_tests_step_counter >= delay_steps) & \
                            (self._disturbance_tests_step_counter < delay_steps + duration_steps)
        
        # Create force vector: magnitude * direction vector
        force_magnitudes = self._disturbance_tests_force_magnitudes.clone()
        force_magnitudes = force_magnitudes * should_apply_force.float()
        
        forces_xy = torch.zeros(self.num_envs, 3, device=self.device)
        forces_xy[:, 0] = force_magnitudes * self._disturbance_tests_direction_x
        forces_xy[:, 1] = force_magnitudes * self._disturbance_tests_direction_y
        # z component is 0
        
        # Apply forces to root body
        forces = forces_xy.unsqueeze(1)  # (num_envs, 1, 3)
        torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        # Get base position for force application point (apply at base center)
        base_pos = self.robot.data.root_pos_w
        positions = base_pos.unsqueeze(1)  # (num_envs, 1, 3)
        
        self.robot.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            positions=positions,
            body_ids=[0],  # Apply to root body
            is_global=True,
        )
        self.robot.write_data_to_sim()
        
        # for environments 0, 99, 999, 9999 print debug info about the applied force
        for env_idx in [0, 99, 999, 9999]:
            if should_apply_force[env_idx]:
                force_env = forces[env_idx, 0].cpu().numpy()
                print(f"[DEBUG] Env {env_idx}: Disturbance TEST FORCE APPLIED - step {self._disturbance_tests_step_counter}, "
                      f"magnitude={force_magnitudes[env_idx].item():.4f}N, direction=({self._disturbance_tests_direction_x:.4f}, {self._disturbance_tests_direction_y:.4f}), "
                      f"force_vector={force_env}")

    def save_disturbance_tests_results(self, log_dir: str, direction: float) -> None:
        """Save disturbance tests results to CSV file.
        
        CSV contains: environment_id, force_magnitude, survived
        """
        import csv
        import os
        
        csv_filename = os.path.join(log_dir, f"disturbance_tests_direction_{direction:.1f}deg.csv")
        
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['env_id', 'force_magnitude_N', 'survived'])
            
            for env_id in range(self.num_envs):
                force_magnitude = self._disturbance_tests_force_magnitudes[env_id].item()
                survived = self._disturbance_tests_survived[env_id].item()
                writer.writerow([env_id, force_magnitude, survived])
        
        print(f"[INFO] Disturbance tests results saved to: {csv_filename}")

    def add_noise_to_observations(self, base_ang_vel, joint_pos_act, joint_vel_act, roll_pitch):
        noise = torch.randn_like(joint_pos_act, device=self.device)
        noise_std = torch.tensor(
            self.cfg.joint_pos_noise_std,
            device=self.device,
            dtype=joint_pos_act.dtype,
        )
        joint_pos_act = (
            joint_pos_act + noise * noise_std
        )  # this is element-wise multiplication, the dot product would be done by the @ operator and would be incorrect here

        # Joint velocity noise
        noise = torch.randn_like(joint_vel_act, device=self.device)
        noise_std = torch.tensor(
            self.cfg.joint_vel_noise_std,
            device=self.device,
            dtype=joint_vel_act.dtype,
        )
        joint_vel_act = joint_vel_act + noise * noise_std

        # Add noise to roll and pitch Euler angles
        noise = torch.randn_like(roll_pitch, device=self.device)
        roll_pitch = roll_pitch + noise * self.cfg.base_rp_noise_std

        # Base angular velocity noise
        noise = torch.randn_like(base_ang_vel, device=self.device)
        base_ang_vel = base_ang_vel + noise * self.cfg.base_ang_vel_noise_std

        return base_ang_vel, joint_pos_act, joint_vel_act, roll_pitch

    def compute_actuator_dynamics(self, tau_m: torch.Tensor, omega_l: torch.Tensor) -> torch.Tensor:
        """Apply series elastic actuator (SEA) dynamics to motor torque commands.
        
        Computes the actual joint torques after passing through the actuator dynamics model
        with state-space representations for motor and spring/elastic subsystems.
        
        State equations:
            x_H[k+1] = A_H * x_H[k] + B_H * tau_m[k]  (motor-side dynamics)
            x_Z[k+1] = A_Z * x_Z[k] + B_Z * omega_l[k]  (spring-side dynamics)
            tau_k[k] = C_H * x_H[k] + D_H * tau_m[k] - (C_Z * x_Z[k] + D_Z * omega_l[k])
        
        Args:
            tau_m: Motor torque commands, shape (num_envs, 8)
            omega_l: Load-side joint velocities, shape (num_envs, 8)
            
        Returns:
            tau_k: Actual joint torques applied to the simulation, shape (num_envs, 8)
        """
        if not self.cfg.apply_actuator_dynamics:
            return tau_m
        
        # Get model type for each joint (0=knee, 1=leg)
        model_types = self._sea_joint_to_model  # shape (8,)
        
        # Compute H system: x_H[k+1] = A_H * x_H[k] + B_H * tau_m[k]
        # Process both model types
        x_H_new = self._sea_x_H.clone()
        for model_type, model_name in [(0, 'knee'), (1, 'leg')]:
            mask = (model_types == model_type).nonzero(as_tuple=False).squeeze(-1)
            if len(mask) > 0:
                A_H = self._sea_models[model_name]['H']['A']  # (2, 2)
                B_H = self._sea_models[model_name]['H']['B']  # (2, 1)
                
                # Get states and inputs for this model type
                x_H_masked = self._sea_x_H[:, mask, :]  # (num_envs, num_joints_of_type, 2)
                tau_m_masked = tau_m[:, mask]  # (num_envs, num_joints_of_type)
                
                # Compute next state
                # x_H[k+1] = A_H @ x_H[k] + B_H @ tau_m[k]
                # Vectorized: (num_envs, num_joints_of_type, 2) = (2,2) @ (num_envs, num_joints_of_type, 2) + (2,1) @ (num_envs, num_joints_of_type, 1)
                x_H_next = torch.einsum('ij,nkj->nki', A_H, x_H_masked) + torch.einsum('i,nk->nki', B_H.squeeze(-1), tau_m_masked)
                x_H_new[:, mask, :] = x_H_next
        
        # Compute Z system: x_Z[k+1] = A_Z * x_Z[k] + B_Z * omega_l[k]
        x_Z_new = self._sea_x_Z.clone()
        for model_type, model_name in [(0, 'knee'), (1, 'leg')]:
            mask = (model_types == model_type).nonzero(as_tuple=False).squeeze(-1)
            if len(mask) > 0:
                A_Z = self._sea_models[model_name]['Z']['A']  # (2, 2)
                B_Z = self._sea_models[model_name]['Z']['B']  # (2, 1)
                
                # Get states and inputs for this model type
                x_Z_masked = self._sea_x_Z[:, mask, :]  # (num_envs, num_joints_of_type, 2)
                omega_l_masked = omega_l[:, mask]  # (num_envs, num_joints_of_type)
                
                # Compute next state
                x_Z_next = torch.einsum('ij,nkj->nki', A_Z, x_Z_masked) + torch.einsum('i,nk->nki', B_Z.squeeze(-1), omega_l_masked)
                x_Z_new[:, mask, :] = x_Z_next
        
        # Update state buffers for next timestep
        self._sea_x_H = x_H_new
        self._sea_x_Z = x_Z_new
        
        # Compute output torques: tau_k = C_H * x_H + D_H * tau_m - (C_Z * x_Z + D_Z * omega_l)
        tau_k = torch.zeros_like(tau_m)
        for model_type, model_name in [(0, 'knee'), (1, 'leg')]:
            mask = (model_types == model_type).nonzero(as_tuple=False).squeeze(-1)
            if len(mask) > 0:
                C_H = self._sea_models[model_name]['H']['C']  # (1, 2)
                D_H = self._sea_models[model_name]['H']['D']  # (1, 1)
                C_Z = self._sea_models[model_name]['Z']['C']  # (1, 2)
                D_Z = self._sea_models[model_name]['Z']['D']  # (1, 1)
                
                # Get states and inputs
                x_H_masked = self._sea_x_H[:, mask, :]  # (num_envs, num_joints_of_type, 2)
                x_Z_masked = self._sea_x_Z[:, mask, :]  # (num_envs, num_joints_of_type, 2)
                tau_m_masked = tau_m[:, mask]  # (num_envs, num_joints_of_type)
                omega_l_masked = omega_l[:, mask]  # (num_envs, num_joints_of_type)
                
                # Compute H output: C_H * x_H + D_H * tau_m
                tau_H = torch.einsum('ij,nkj->nk', C_H, x_H_masked) + D_H.squeeze() * tau_m_masked
                
                # Compute Z output: C_Z * x_Z + D_Z * omega_l
                tau_Z = torch.einsum('ij,nkj->nk', C_Z, x_Z_masked) + D_Z.squeeze() * omega_l_masked
                
                # Final torque: tau_k = tau_H - tau_Z  (note the minus sign!)
                tau_k[:, mask] = tau_H - tau_Z
        
        return tau_k

    # ------- Observation preprocessing and normalization -------

    def preprocess_joint_pos_act(self, joint_pos_act: torch.Tensor) -> torch.Tensor:
        """Normalize joint positions to [-1, 1] using configured bounds."""
        joint_pos_min = torch.tensor(
            self.cfg.obs_joint_pos_min, device=self.device, dtype=torch.float32
        )
        joint_pos_max = torch.tensor(
            self.cfg.obs_joint_pos_max, device=self.device, dtype=torch.float32
        )
        # Clamp to bounds
        joint_pos_clamped = torch.clamp(joint_pos_act, joint_pos_min, joint_pos_max)
        # Normalize to [-1, 1]
        mid_point = (joint_pos_min + joint_pos_max) / 2.0
        range_val = (joint_pos_max - joint_pos_min) / 2.0
        joint_pos_norm = (joint_pos_clamped - mid_point) / range_val
        return joint_pos_norm

    def preprocess_joint_vel_act(self, joint_vel_act: torch.Tensor) -> torch.Tensor:
        """Normalize joint velocities to [-1, 1] using symmetric bounds."""
        # Clamp to [-bound, +bound]
        joint_vel_clamped = torch.clamp(
            joint_vel_act, 
            -self.cfg.obs_joint_vel_bound, 
            self.cfg.obs_joint_vel_bound
        )
        # Normalize to [-1, 1]
        joint_vel_norm = joint_vel_clamped / self.cfg.obs_joint_vel_bound
        return joint_vel_norm

    def preprocess_roll_pitch(self, roll_pitch: torch.Tensor) -> torch.Tensor:
        """Normalize roll and pitch angles to [-1, 1] using symmetric bounds."""
        # Clamp to [-bound, +bound]
        roll_pitch_clamped = torch.clamp(
            roll_pitch,
            -self.cfg.obs_roll_pitch_bound,
            self.cfg.obs_roll_pitch_bound
        )
        # Normalize to [-1, 1]
        roll_pitch_norm = roll_pitch_clamped / self.cfg.obs_roll_pitch_bound
        return roll_pitch_norm

    def preprocess_base_ang_vel(self, base_ang_vel: torch.Tensor) -> torch.Tensor:
        """Normalize base angular velocity to [-1, 1] using symmetric bounds."""
        # Clamp to [-bound, +bound]
        base_ang_vel_clamped = torch.clamp(
            base_ang_vel,
            -self.cfg.obs_base_ang_vel_bound,
            self.cfg.obs_base_ang_vel_bound
        )
        # Normalize to [-1, 1]
        base_ang_vel_norm = base_ang_vel_clamped / self.cfg.obs_base_ang_vel_bound
        return base_ang_vel_norm

    def preprocess_walk_speeds(self, speed_x: torch.Tensor, speed_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize walk speed targets to [-1, 1] using symmetric bounds."""
        # Clamp to [-bound, +bound]
        speed_x_clamped = torch.clamp(
            speed_x,
            -self.cfg.obs_walk_speed_bound,
            self.cfg.obs_walk_speed_bound
        )
        speed_y_clamped = torch.clamp(
            speed_y,
            -self.cfg.obs_walk_speed_bound,
            self.cfg.obs_walk_speed_bound
        )
        # Normalize to [-1, 1]
        speed_x_norm = speed_x_clamped / self.cfg.obs_walk_speed_bound
        speed_y_norm = speed_y_clamped / self.cfg.obs_walk_speed_bound
        return speed_x_norm, speed_y_norm