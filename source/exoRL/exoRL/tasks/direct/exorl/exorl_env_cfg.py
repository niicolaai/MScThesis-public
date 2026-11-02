"""
Environment configuration for ExoRL task with exoWE2_V2 robot.
"""
# flake8: noqa: W293

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.mdp import events
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .exoWE2_V2 import EXOWE2_V2_CFG


@configclass
class ExorlEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 10 # control frequency = sim dt * decimation = 1/1000 * 10 = 0.01 s → 100 Hz
    episode_length_s = 10.0

    # action and observation space
    action_scale_position = 1.0  # action scale for position control mode (different from torque control mode)
    use_torque_control_mode = True  # if True, neural network outputs torques instead of joint positions
    action_scale_torque = 100.0  # action scale for torque control mode (separate from position mode)
    action_space = 8
    observation_space = 24  # 8 joint pos + 8 joint vel + 2 roll/pitch + 3 base ang vel + 2 target velocities
    state_space = 1  # not used, but required by DirectRLEnvCfg
    imitation_learning_walking = False  # enable motion imitation learning mode (removes target velocities from observations)

    sim: SimulationCfg = SimulationCfg(dt=1 / 1000, render_interval=decimation)
    robot_cfg = EXOWE2_V2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=200, env_spacing=2.0, replicate_physics=True
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ===== DOMAIN RANDOMIZATION =====
    # Randomize physical properties at episode reset to improve generalization
    @configclass
    class EventCfg:
        """Configuration for MDP events (domain randomization) applied at episode reset."""
        
        # Randomize gravity: gaussian noise with std ±0.2 m/s² on z-component
        reset_gravity = EventTerm(
            func=events.randomize_physics_scene_gravity,
            mode="reset",
            params={
                "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.2]),
                "operation": "add",
                "distribution": "gaussian",
            },
        )
        
        # Randomize robot body masses: ±10% scaling with gaussian distribution
        robot_mass = EventTerm(
            func=events.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="(BasePlate|PelvicAA_R|UpperLeg_R|LowerLeg_R|AnkleFlex_R|AnkleAbd_R|PelvicAA_L|UpperLeg_L|LowerLeg_L|AnkleFlex_L|AnkleAbd_L)"), # TODO NH: is randomizing the mass also randomizing the inertia? if not, should we randomize inertia as well?
                "mass_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        
        # Randomize robot material properties (friction): ±30%
        robot_physics_material = EventTerm(
            func=events.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="AnkleAbd_L|AnkleAbd_R"),  # only randomize foot-ground friction, not friction of other robot bodies
                "static_friction_range": (0.7, 1.3),
                "dynamic_friction_range": (0.7, 1.3),
                "restitution_range": (1.0, 1.0),  # "bounciness" of the material, keep at 1.0 (no energy loss) to avoid instability
                "num_buckets": 250,  # a meterial bucket is a tuple of (static_friction, dynamic_friction, restitution), so 250 different combinations of these properties will be generated and assigned to robot bodies for randomization
            },
        )

    events: EventCfg = EventCfg()

    # reward scales, two sets of parameters for imitation learning vs. non-imitation learning mode
    if imitation_learning_walking: # imitation learning
        orientation_penalty_scale = 0.0
        height_reward_scale = 0.0
        action_change_penalty_scale_position = 0*5e-2
        action_change_penalty_scale_torque = 0*3e-7
        alive_reward = 0.1
        early_termination_penalty_scale = 10.0
        joint_symmetry_penalty_scale = 0.0
        cbf_angle_reward_scale = 0*1e-3
        cbf_torque_reward_scale = 0*4e-4
        walk_velocity_reward_scale = 0.0
        spawn_position_reward_scale = 0.0
        imitation_q_reward_scale = 1.0
        imitation_dq_reward_scale = 0*1e-2
        imitation_pelvis_reward_scale = 5.0
    else: # standing balance
        orientation_penalty_scale = 3.0
        height_reward_scale = 10.0
        action_change_penalty_scale_position = 5e-2
        action_change_penalty_scale_torque = 3e-7
        alive_reward = 0.1
        early_termination_penalty_scale = 10.0
        joint_symmetry_penalty_scale = 0.3
        cbf_angle_reward_scale = 1e-3
        cbf_torque_reward_scale = 4e-4 # was 4e-5
        walk_velocity_reward_scale = 0.0 # 5.0
        spawn_position_reward_scale = 0.0 # 1.0
        imitation_q_reward_scale = 0.0  # was 100.0
        imitation_dq_reward_scale = 0.0  # was 1e-2
        imitation_pelvis_reward_scale = 0.0

    # Control barrier function parameters for joint angle constraints
    cbf_angle_delta_a = 1.0  # Scales the overall magnitude of the barrier reward
    cbf_angle_delta_b = 1.0  # Controls sharpness of the drop-off outside bounds
    cbf_angle_gamma = 1.0
    cbf_angle_beta = 0 # scales the derivative term

    # Control barrier function parameters for torque limits
    cbf_torque_gamma = 100
    cbf_torque_beta = 0 # scales the derivative term
    joint_torque_saturation = 100.0
    cbf_torque_delta_a = 1.0 / (joint_torque_saturation**2)  # → maximum value of h is 1 when torque is at 0

    # walking velocity tracking parameters
    target_walk_speed_x = 0.0  # target walking speed in x direction (m/s)
    target_walk_speed_y = 0.0  # target walking speed in y direction (m/s)

    # Joint angle constraints for all 8 actuated joints (in degrees, then converted to radians)
    # 8 actuated joints: HipAA_L, HipAA_R, HipFE_L, HipFE_R, KneeFE_L, KneeFE_R, AnkleFE_L, AnkleFE_R
    joint_angle_min_constraint = [
        -0.3,  # HipAA_L: Adduction limit
        -0.3,  # HipAA_R
        -0.7,  # HipFE_L: Extension limit
        -0.7,  # HipFE_R
        0.0,    # KneeFE_L: Full extension
        0.0,    # KneeFE_R
        -0.65,  # AnkleFE_L: Plantarflexion limit
        -0.65,  # AnkleFE_R
    ]
    joint_angle_max_constraint = [
        0.35,   # HipAA_L: Abduction limit
        0.35,   # HipAA_R
        2.0,  # HipFE_L: Flexion limit
        2.0,  # HipFE_R
        2.1,    # KneeFE_L: Flexion limit
        2.1,    # KneeFE_R
        0.65,   # AnkleFE_L: Dorsiflexion limit
        0.65,   # AnkleFE_R
    ]

    target_height = 1.75  # for some reason the base pos is offset by ~1m, so imagine this to be goal height of 0.75m (typical human CoM height) instead of 1.75m

    # Observation normalization bounds
    # Note: joint position bounds use the constraint limits (asymmetric), others are symmetric
    # obs_joint_pos_min = [-0.3, -0.3, -0.7, -0.7, 0.0, 0.0, -0.65, -0.65]
    # obs_joint_pos_max = [0.35, 0.35, 2.0, 2.0, 2.1, 2.1, 0.65, 0.65]
    obs_joint_pos_min = [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]
    obs_joint_pos_max = [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]
    obs_joint_vel_bound = 50.0  # rad/s, symmetric bounds: [-50, +50]
    obs_roll_pitch_bound = np.pi / 2  # rad (90 degrees), symmetric bounds: [-π/2, +π/2]
    obs_base_ang_vel_bound = 20.0  # rad/s, symmetric bounds: [-20, +20]
    obs_walk_speed_bound = 2.5  # m/s, symmetric bounds: [-2.5, +2.5]    obs_pelvis_vel_bound = 2.0  # m/s, symmetric bounds: [-2, +2], used for pelvis velocity from motion data
    # logging parameters
    log_sub_rewards = False
    log_com_position = True  # log center of mass position of env 0 to CSV file

    # termination conditions
    fall_height_threshold = 1.3  # height below which we consider the robot fallen

    # noise parameters (standard deviation for Gaussian noise)
    apply_sensor_noise = False
    joint_pos_noise_std = [
        1e-5,
        1e-5,
        1e-5,
        1e-5,
        1e-5,
        1e-5,
        1e-5,
        1e-5,
    ]
    joint_vel_noise_std = [
        1e-2,
        1e-2,
        1e-2,
        1e-2,
        1e-2,
        1e-2,
        1e-2,
        1e-2,
    ]
    base_rp_noise_std = 0.001  # rad
    base_lin_acc_noise_std = 0.01  # m/s^2
    base_ang_vel_noise_std = 0.001  # rad/s

    # joint angle limits for randomized reset (in degrees, then converted to radians)
    # 8 actuated joints: HipAA_L, HipAA_R, HipFE_L, HipFE_R, KneeFE_L, KneeFE_R, AnkleFE_L, AnkleFE_R
    # Passive joints: AnkleAA_L, AnkleAA_R (indices 8-9)
    apply_configuration_randomization = False

    _joint_angle_min_deg = [
        0.0,  # HipAA_L: Abduction/Adduction
        0.0,  # HipAA_R
        -30.0,  # HipFE_L: Flexion/Extension
        -30.0,  # HipFE_R
        0.0,  # KneeFE_L: Flexion/Extension
        0.0,  # KneeFE_R
        -20.0,  # AnkleFE_L: Flexion/Extension
        -20.0,  # AnkleFE_R
    ]
    joint_angle_min_rad = [d * np.pi / 180 for d in _joint_angle_min_deg]

    _joint_angle_max_deg = [
        20.0,  # HipAA_L
        20.0,  # HipAA_R
        10.0,  # HipFE_L
        10.0,  # HipFE_R
        80.0,  # KneeFE_L
        80.0,  # KneeFE_R
        20.0,  # AnkleFE_L
        20.0,  # AnkleFE_R
    ]
    joint_angle_max_rad = [d * np.pi / 180 for d in _joint_angle_max_deg]

    # Per-joint standard deviation for randomization around default configuration
    joint_angle_config_randomization_std_rad = [
        0.1,    # HipAA_L: ~5.7 deg
        0.1,    # HipAA_R: ~5.7 deg
        0.2,    # HipFE_L: ~11.5 deg
        0.2,    # HipFE_R: ~11.5 deg
        0.2,    # KneeFE_L: ~11.5 deg
        0.2,    # KneeFE_R: ~11.5 deg
        0.15,   # AnkleFE_L: ~8.6 deg
        0.15,   # AnkleFE_R: ~8.6 deg
    ]

    # Actuator dynamics model parameters (Series Elastic Actuator)
    # Two models: Knee (LK, RK) and Leg (LHA, LHF, LA, RHA, RHF, RA)
    # Joint indices: 0-3, 6-7 → Leg model; 4-5 → Knee model
    apply_actuator_dynamics = False

    # Knee model state-space matrices for H and Z systems
    # State: [state1, state2]
    # Input: Motor torque (H system) or load-side velocity (Z system)
    # Output: Torque
    sea_knee_H_A = [
            [1.995240840484988, -0.9980033778908122],
            [1, 0],
        ]
    sea_knee_H_B = [
            [0.0625],
            [0],
        ]
    sea_knee_H_C = [
            [0.04414800906847185, 2.206297304771163e-05],
        ]
    sea_knee_H_D = [
            [0.0006906343514561544],
        ]

    sea_knee_Z_A = [
            [1.995240840484988, -0.9980033778908122],
            [1, 0],
        ]
    sea_knee_Z_B = [
            [2],
            [0],
        ]
    sea_knee_Z_C = [
            [0.7654121108600733, -0.7649394031165713],
        ]
    sea_knee_Z_D = [
            [0.7664702834524332],
        ]

    # Leg model state-space matrices for H and Z systems
    sea_leg_H_A = [
            [1.99644776439489, -0.9980027747304374],
            [1, 0],
        ]
    sea_leg_H_B = [
            [0.03125],
            [0],
        ]
    sea_leg_H_C = [
            [0.04971614063287835, 2.484564749269218e-05],
        ]
    sea_leg_H_D = [
            [0.0003887525838868775],
        ]

    sea_leg_Z_A = [
            [1.99644776439489, -0.9980027747304374],
            [1, 0],
        ]
    sea_leg_Z_B = [
            [2],
            [0],
        ]
    sea_leg_Z_C = [
            [0.7661060098952616, -0.7651702527459607],
        ]
    sea_leg_Z_D = [
            [0.7667018267681587],
        ]

    # disturbance force parameters
    apply_disturbance_forces = False
    disturbance_cycle_interval = 300  # number of cycles before applying next disturbance
    disturbance_duration = 40  # number of cycles to apply disturbance force (ramp-in 10 + steady 20 + ramp-out 10)
    disturbance_force_min = 1.0  # minimum force magnitude in Newtons
    disturbance_force_max = 20.0  # maximum force magnitude in Newtons
    disturbance_com_offset_range = 0.2  # max offset from center of mass in meters (0 = always at CoM, >0 = random offset up to this value)
    disturbance_ramp_in_steps = 10  # number of steps to ramp up force from 0 to max (100ms for smooth initialization)
    disturbance_ramp_out_steps = 10  # number of steps to ramp down force from max to 0 (100ms for smooth termination)

    # disturbance tests parameters (special test mode)
    disturbance_tests_mode = False  # Enable disturbance tests mode (10000 envs, single episode)
    disturbance_tests_direction = 0.0  # Direction angle in degrees for disturbance force in xy plane
    disturbance_tests_delay_steps = 300  # number of steps to wait before applying force (3 seconds at 100 Hz)
    disturbance_tests_force_duration_steps = 15  # number of steps to apply force (150 ms at 100 Hz)
    disturbance_tests_max_force = 500.0 * 9999.0 / 10000.0  # maximum force magnitude in Newtons
