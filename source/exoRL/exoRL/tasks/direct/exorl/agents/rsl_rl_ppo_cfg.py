# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# flake8: noqa: W293

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # num_steps_per_env: number of steps collected per environment before updating policy
    num_steps_per_env = 128

    # max_iterations: total number of policy updates to perform during training, is overwritten by command line argument --max_iterations, so it doesn't really matter what this is set to here
    max_iterations = 5000

    # save_interval: how often to save policy checkpoint (every N episode updates)
    save_interval = 500

    experiment_name = "exorl_ppo_experiment"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # was 0.005
        num_learning_epochs=4,
        num_mini_batches=8,
        learning_rate=2.0e-3,  # was 5.0e-4
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }
