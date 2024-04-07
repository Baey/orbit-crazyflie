# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This is a example script for crazyflie drone simulation developement. 
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This is a example script for crazyflie drone simulation developement.")
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

# Pre-defined configs
from omni.isaac.orbit_assets.crazyflie import CRAZYFLIE_CFG


class CrazyfliesV1(Articulation):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Parameters
        self._rotor_constant = 5.54858e-6
        self._rolling_moment_coefficient = 1.24e-7

    def reset(self):
        joint_pos, joint_vel = self.get_position(), self.get_velocity()
        super().write_joint_state_to_sim(joint_pos, joint_vel)
        super().write_root_pose_to_sim(super().data.default_root_state[:, :7])
        super().write_root_velocity_to_sim(super().data.default_root_state[:, 7:])
        super().reset()

    def apply_changes(self):
        super().write_data_to_sim()

    def update(self, dt):
        super().update(dt)

    def get_drones_count(self):
        return super().num_instances

    #TODO: get position using sesors
    def get_position(self):
        return super().data.joint_pos.clone()
    
    #TODO: get velocity using sesors
    def get_velocity(self):
        return super().data.joint_vel.clone()
    
    #TODO: get acceleration using sesors
    def get_acceleration(self):
        return super().data.joint_acc.clone()

    def set_effort(self, efforts: torch.tensor):
        super().set_joint_effort_target(efforts)
        velocity = self.get_velocity()

        drones_count = self.get_drones_count()

        rolling_moment = torch.sum(self._rolling_moment_coefficient * torch.square(velocity))
        torques = torch.zeros(drones_count, 4, 3)
        #FIXME apply non-zero torques
        # torques[:, :, 2] = rolling_moment

        forces = torch.zeros(drones_count, 4, 3)
        forces[:, :, 2] = self._rotor_constant * torch.square(velocity)
        super().set_external_force_and_torque(forces=forces, torques=torques, body_ids=[1, 2, 3, 4])


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
    drone_cfg.spawn.func("/World/Crazyflie/Drone1", drone_cfg.spawn, translation=(1.5, 0.5, 0.42))
    # create handles for the robots
    drones = CrazyfliesV1(drone_cfg.replace(prim_path="/World/Crazyflie/Drone.*"))

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
        if count % 300 == 0:
            # reset counters
            sim_time = 0.0
            count = 0

            # reset drones state
            drones.reset()
            print(">>>>>>>> Reset!")

        # set efforts
        efforts = torch.tensor([1, -1, 1, -1], dtype=torch.float32) * 3.8e-3
        drones.set_effort(efforts)

        # apply changes
        drones.apply_changes()

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
