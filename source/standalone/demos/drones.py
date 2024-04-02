# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate a bipedal robot.

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a bipedal robot.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.sim import SimulationContext

##
# Pre-defined configs
##
from omni.isaac.orbit_assets.crazyflie import CRAZYFLIE_CFG  # isort:skip


def main():
    """Main function."""

    # Load kit helper
    sim = SimulationContext(
        sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False, dt=0.005, physx=sim_utils.PhysxCfg(use_gpu=False))
    )
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Robots
    drone_cfg = CRAZYFLIE_CFG
    drone_cfg.spawn.func("/World/Crazyflie/Drone_1", drone_cfg.spawn, translation=(1.5, 0.5, 0.42))

    # create handles for the robots
    drones = Articulation(drone_cfg.replace(prim_path="/World/Crazyflie/Drone.*"))

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 2000 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset dof state
            joint_pos, joint_vel = drones.data.default_joint_pos, drones.data.default_joint_vel
            drones.write_joint_state_to_sim(joint_pos, joint_vel)
            drones.write_root_pose_to_sim(drones.data.default_root_state[:, :7])
            drones.write_root_velocity_to_sim(drones.data.default_root_state[:, 7:])
            drones.reset()
            # reset command
            print(">>>>>>>> Reset!")
        # apply action to the robot
        velocities = torch.ones(1, 1) * 1000.0
        drones.set_joint_velocity_target(velocities, joint_ids=[1])
        # forces = torch.tensor([0.0, 0.0, 0.00]).reshape(1, 3)
        # drones.set_external_force_and_torque(forces=forces, torques=torch.zeros(1,3), body_ids=[1, 2, 3, 4])
        drones.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        drones.update(sim_dt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
