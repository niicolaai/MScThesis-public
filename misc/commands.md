# Isaaclab
Starting/entering/stopping container (execute from within docker subdirectory)
```shell
python container.py start/enter/stop
```

Copy file to docker container
```shell
docker cp <path_to_robot_assets> <container_id>:/workspace/
```

Start training
```shell
/workspace/isaaclab/isaaclab.sh -p ../../scripts/rsl_rl/train.py --task=Template-Exorl-Direct-v0 --headless
```

Disturbance tests:
PD policy:
```shell
/workspace/isaaclab/isaaclab.sh -p ../../scripts/rsl_rl/play.py --task=Template-Exorl-Direct-v0 --num_envs=10000 --checkpoint=/exoWE2/exoRL/source/exoRL/logs/rsl_rl/exorl_ppo_experiment/2026-04-26_02-21-34/model_1747.pt --disturbance_tests --direction 0.0
```

Torque policy:
```shell
/workspace/isaaclab/isaaclab.sh -p ../../scripts/rsl_rl/play.py --task=Template-Exorl-Direct-v0 --num_envs=10000 --checkpoint=/exoWE2/exoRL/source/exoRL/logs/rsl_rl/exorl_ppo_experiment/2026-04-25_14-37-04/model_1747.pt --disturbance_tests --direction 0.0
```

Names:
```
? Task type: External
? Project path: /exoWE2
? Project name: exoskeleton_RL
```

# Other handy commands
Use VSCode
```shell
~/VSCode-linux-x64/bin/code
```

Update .ipynb file changes
```shell
git add --renormalize '*.ipynb'
```
