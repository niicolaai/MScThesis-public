# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# flake8: noqa: W293

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import numpy as np

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
parser.add_argument(
    "--sensor_noise",
    action="store_true",
    default=False,
    help="Enable sensor noise in observations.",
)
parser.add_argument(
    "--configuration_randomization",
    action="store_true",
    default=False,
    help="Enable configuration randomization.",
)
parser.add_argument(
    "--disturbance_forces",
    action="store_true",
    default=False,
    help="Enable disturbance forces during training.",
)
parser.add_argument(
    "--log_sub_rewards",
    action="store_true",
    default=False,
    help="Enable logging of individual reward components to file.",
)
parser.add_argument(
    "--log_com_position",
    action="store_true",
    default=False,
    help="Enable logging of center of mass position for environment 0.",
)
parser.add_argument(
    "--walk_speed_x",
    type=float,
    default=None,
    help="Target walking speed in x direction (m/s).",
)
parser.add_argument(
    "--walk_speed_y",
    type=float,
    default=None,
    help="Target walking speed in y direction (m/s).",
)
parser.add_argument(
    "--imitation_learning_walking",
    action="store_true",
    default=False,
    help="Enable motion imitation learning from motion_data.npz dataset.",
)
parser.add_argument(
    "--disturbance_tests",
    action="store_true",
    default=False,
    help="Enable disturbance tests mode (10000 envs, single episode, force direction based on --direction).",
)
parser.add_argument(
    "--direction",
    type=float,
    default=0.0,
    help="Direction angle in degrees for disturbance force in xy plane (0-360).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import exoRL.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    
    # Handle disturbance tests mode
    if args_cli.disturbance_tests:
        env_cfg.scene.num_envs = 10000
        env_cfg.apply_configuration_randomization = False
        env_cfg.apply_sensor_noise = True  # Keep sensor noise on for disturbance tests
        env_cfg.disturbance_tests_mode = True
        env_cfg.disturbance_tests_direction = args_cli.direction
    else:
        env_cfg.scene.num_envs = (
            args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        )

    if args_cli.sensor_noise:
        env_cfg.apply_sensor_noise = args_cli.sensor_noise
    if args_cli.disturbance_forces:
        env_cfg.apply_disturbance_forces = args_cli.disturbance_forces
    if args_cli.configuration_randomization:
        env_cfg.apply_configuration_randomization = args_cli.configuration_randomization
    if args_cli.log_sub_rewards:
        env_cfg.log_sub_rewards = args_cli.log_sub_rewards
    if args_cli.log_com_position:
        env_cfg.log_com_position = args_cli.log_com_position
    if args_cli.walk_speed_x is not None:
        env_cfg.target_walk_speed_x = args_cli.walk_speed_x
    if args_cli.walk_speed_y is not None:
        env_cfg.target_walk_speed_y = args_cli.walk_speed_y
    if args_cli.imitation_learning_walking:
        env_cfg.imitation_learning_walking = args_cli.imitation_learning_walking

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print(
                "[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task."
            )
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # set camera view for recording videos at a good angle
    env.unwrapped.sim.set_camera_view(
        eye=[2.5, 2.5, 4.0],
        target=[0.0, 0.0, 1.2],
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx"
    )

    logged_data = {"observations": [], "actions": [], "rewards": [], "dones": []}

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # In disturbance tests mode, run for exactly one episode
    max_steps_for_disturbance_tests = None
    if args_cli.disturbance_tests:
        # Calculate max steps: episode_length_s / (sim_dt * decimation)
        # With episode_length_s=10, sim_dt=0.001, decimation=10: 10 / (0.001 * 10) = 1000 steps
        max_steps_for_disturbance_tests = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
        print(f"[INFO] Disturbance tests mode: Running for {max_steps_for_disturbance_tests} steps (~{env_cfg.episode_length_s}s)")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # temporarily set actions of hip FE to -1
            # actions[:, 2] = 1.0  # HipFE_L
            # actions[:, 3] = 1.0  # HipFE_R
            # env stepping
            obs, rewards, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

            # Log data
            logged_data["observations"].append(obs.cpu().numpy())
            logged_data["actions"].append(actions.cpu().numpy())
            logged_data["rewards"].append(rewards.cpu().numpy())
            logged_data["dones"].append(dones.cpu().numpy())

        timestep += 1
        
        # Check if we should exit (for disturbance tests or video recording)
        should_exit = False
        if args_cli.disturbance_tests and timestep >= max_steps_for_disturbance_tests:
            print(f"[INFO] Disturbance tests completed ({timestep} steps), exiting...")
            should_exit = True
        elif args_cli.video:
            if timestep == args_cli.video_length:
                # Save logged data to NPZ
                try:
                    log_file = os.path.join(log_dir, "play_data.npz")
                    np.savez_compressed(
                        log_file,
                        **{
                            k: np.array(v, dtype=object) for k, v in logged_data.items()
                        },
                    )
                    print(f"[INFO] Data logged to: {log_file}")
                except Exception as e:
                    print(f"[WARN] Failed to save logged data: {e}")
                should_exit = True
        
        if should_exit:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # save sub-rewards log if enabled
    if env_cfg.log_sub_rewards:
        env.unwrapped.save_sub_rewards(log_dir)

    # save center of mass position log if enabled
    if env_cfg.log_com_position:
        env.unwrapped.save_com_position(log_dir)

    # save disturbance tests results if enabled
    if args_cli.disturbance_tests:
        env.unwrapped.save_disturbance_tests_results(log_dir, args_cli.direction)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
