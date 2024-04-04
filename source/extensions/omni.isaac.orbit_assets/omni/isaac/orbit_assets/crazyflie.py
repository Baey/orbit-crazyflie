from __future__ import annotations

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import DCMotorCfg
from omni.isaac.orbit.assets.articulation import ArticulationCfg
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##


CRAZYFLIE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Crazyflie/cf2x.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.2,
            angular_damping=0.2,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0.0, 0.0, 0.0),
        joint_pos={
            "m1_joint": 0.0,
            "m2_joint": 0.0,
            "m3_joint": 0.0,
            "m4_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "rotors": DCMotorCfg(
            joint_names_expr=[".*_joint"],
            effort_limit=1.0,
            friction=0.0,
            velocity_limit=231500.0,
            stiffness={
                ".*_joint": 0.0,
            },
            damping={
                ".*_joint": 1.0,
            },
            saturation_effort=8e-3
        )
    },
)