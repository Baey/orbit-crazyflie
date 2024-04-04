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
from copy import copy

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
    sim.set_camera_view(eye=(1.0, 1.0, 1.0), target=(0.0, 0.0, 0.0))

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
    # sp = 62.8
    rotor_constant = 5.54858e-6
    rolling_moment_coefficient = 1.24e-7
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 300 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset dof state
            joint_pos, joint_vel = drones.data.default_joint_pos, drones.data.default_joint_vel
            drones.write_joint_state_to_sim(joint_pos, joint_vel)
            drones.write_root_pose_to_sim(drones.data.default_root_state[:, :7])
            drones.write_root_velocity_to_sim(drones.data.default_root_state[:, 7:])
            drones.reset()
            # velocities_sp = torch.tensor([1, -1, 1, -1], dtype=torch.float32) * sp
            # drones.set_joint_velocity_target(velocities_sp)
            # reset command
            print(">>>>>>>> Reset!")
        # apply action to the robot
        # velocities_sp = torch.tensor([1, -1, 1, -1], dtype=torch.float32) * torch.clamp(torch.tensor([count / 10]), 0.0, 240.0)
        efforts = torch.tensor([1, -1, 1, -1], dtype=torch.float32) * 3.8e-3
        drones.set_joint_effort_target(efforts)
        # drones.set_joint_velocity_target(velocities_sp)
        velocities_current = drones.data.joint_vel.clone()
        forces = torch.zeros(1, 4, 3)
        forces[:, :, 2] = rotor_constant * torch.square(velocities_current)
        rolling_moment = torch.sum(rolling_moment_coefficient * torch.square(velocities_current))
        torques = torch.zeros(1, 1, 3)
        torques[:, :, 2] = rolling_moment
        drones.set_external_force_and_torque(forces=forces, torques=torch.zeros(1, 4, 3), body_ids=[1, 2, 3, 4])
        # drones.set_external_force_and_torque(forces=torch.zeros(1, 1, 3), torques=torques, body_ids=[0])

        print(velocities_current)
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
