"""Configuration for the WE2 Exoskeleton robot."""
# flake8: noqa: W293

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg


EXOWE2_V2_CFG = ArticulationCfg(
    # prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/exoWE2/mujoco_assets_V2/WE2_3D_phys_Floor_newLimits.usd",  # WE2_3D_phys_Floor.usd = version with foot markers, WE2_3D_phys_Floor_NoFootMarkers.usd = version without foot markers
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            "HipAA_LV2": 0.0111,
            "HipAA_RV2": 0.0111,
            "HipFE_L": -0.1310,
            "HipFE_R": -0.1310,
            "KneeFE_L": 0.3042,
            "KneeFE_R": 0.3042,
            "AnkleFE_L": -0.35,
            "AnkleFE_R": -0.35,
        },
    ),
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                "HipAA_.*",
                "HipFE_.*",
                "KneeFE_.*",
                "AnkleFE_.*",
            ],
            # P and D gains taken from https://doi.org/10.1109/icorr66766.2025.11062929
            stiffness={
                "HipAA_.*": 0*800.0,  # Hip Abduction
                "HipFE_.*": 0*800.0,  # Hip Flexion
                "KneeFE_.*": 0*800.0,  # Knee
                "AnkleFE_.*": 0*800.0,  # Ankle Flexion/Extension
            },
            damping={
                "HipAA_.*": 0*5.0,  # Hip Abduction
                "HipFE_.*": 0*5.0,  # Hip Flexion
                "KneeFE_.*": 0*5.0,  # Knee
                "AnkleFE_.*": 0*5.0,  # Ankle Flexion/Extension
            },
            effort_limit={
                "HipAA_.*": 100.0,
                "HipFE_.*": 100.0,
                "KneeFE_.*": 100.0,
                "AnkleFE_.*": 100.0,
            },
        ),
    },
)
"""Configuration for the WE2 Exoskeleton robot with IdealPD actuators.

This configuration defines an exoskeleton with:
- 10 total joints (8 actuated leg joints + 2 passive ankle abduction/adduction joints)
- IdealPD actuators for 8 leg joints
- Self-collision enabled
- Initial height at 0.95m

Actuated joints (8 total):
- HipAA_L, HipAA_R: Hip Abduction/Adduction
- HipFE_L, HipFE_R: Hip Flexion/Extension
- KneeFE_L, KneeFE_R: Knee Flexion/Extension
- AnkleFE_L, AnkleFE_R: Ankle Flexion/Extension

Passive joints (2 total):
- AnkleAA_L, AnkleAA_R: Ankle Abduction/Adduction

All actuated joints have an effort limit of 100.0 N⋅m.
"""
