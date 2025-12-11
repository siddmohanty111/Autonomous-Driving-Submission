import gymnasium as gym
import numpy as np
import time

from gymnasium import spaces
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
import copy

from highway_env.utils import Vector
from typing import List


class crazy_driver_env(AbstractEnv):
    "Custom highway environment"

    "Agent drives against flow of traffic"
    "agent also avoids NPC cars and cop cars"

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "absolute": False,
                    "normalize": True,
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": True,
                    "lateral": True,
                },
                "lanes_count": 4,
                "vehicles_count": 40,
                "cop_count": 4,
                "duration": 60,  # [s]
                "collision_cop_reward": -10,
                "collision_reward": -4,
                "high_speed_reward": 0.2,
                "reward_speed_range": [20, 30],  # [m/s]
                "offroad_terminal": False,
                "simulation_frequency": 15,  # [Hz]
                "policy_frequency": 3,  # [Hz]
                "screen_width": 1300,  # [px]
                "screen_height": 150,  # [px]
                "centering_position": [0.1, 0.5],
                "scaling": 3,
                "show_trajectories": False,
                "render_agent": True,
                "road_length": 2000,  # [m]
                "agent_start_x": 80,
                "cop_start_x": 40,
                "npc_spawn_min_x": 150,
                "npc_spawn_max_x": 2000,
                "MAX_SPEED": 15,  # [m/s]
                "npc_speed_min": 5,
                "npc_speed_max": 15,
                "cop_respawn_distance": 5.0,
            }
        )

        return config

    def _reset(self):
        self._create_road()
        self._create_vehicles()
        self.copy_of_vehicles = copy.deepcopy(self.road.vehicles)

    def _create_road(self):
        road_net = RoadNetwork()
        lane_width = 4.0
        road_length = self.config["road_length"]

        for i in range(self.config["lanes_count"]):
            road_net.add_lane(
                "a",
                "b",
                StraightLane(
                    [0, i * lane_width],
                    [road_length, i * lane_width],
                    line_types=(
                        LineType.CONTINUOUS_LINE if i == 0 else LineType.STRIPED,
                        (
                            LineType.CONTINUOUS_LINE
                            if i == self.config["lanes_count"] - 1
                            else LineType.STRIPED
                        ),
                    ),
                    width=lane_width,
                ),
            )

        road = Road(
            network=road_net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

        self.road = road

    def _create_vehicles(self):
        # create agent
        agent_start_x = self.config["agent_start_x"]
        ego_vehicle_lane = self.road.network.get_lane(("a", "b", 0))
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_vehicle_lane.position(agent_start_x, 0),
            speed=self.config["MAX_SPEED"] * 0.9,
        )
        ego_vehicle.color = (0, 255, 0)
        ego_vehicle.MAX_SPEED = self.config["MAX_SPEED"]

        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # create npc vehicles
        npc_min_x = self.config["npc_spawn_min_x"]
        npc_max_x = self.config["npc_spawn_max_x"]
        npc_speed_min = self.config["npc_speed_min"]
        npc_speed_max = self.config["npc_speed_max"]

        for i in range(self.config["vehicles_count"]):
            lane_idx = self.road.np_random.integers(0, self.config["lanes_count"])
            lane = self.road.network.get_lane(("a", "b", lane_idx))

            longitudinal = self.road.np_random.uniform(npc_min_x, npc_max_x)
            pick_speed = float(
                self.road.np_random.uniform(npc_speed_min, npc_speed_max)
            )

            v = Vehicle(
                self.road,
                lane.position(longitudinal, 0),
                heading=lane.heading_at(longitudinal),
                speed=-1 * pick_speed,
            )

            v.enable_lane_change = False

            setattr(v, "picked_speed", pick_speed)
            self.road.vehicles.append(v)

        # police cars
        cop_start_x = self.config["cop_start_x"]
        for lane_idx in range(self.config["cop_count"]):
            lane = self.road.network.get_lane(("a", "b", lane_idx))
            cop_position = cop_start_x

            cop = Cop(
                road=self.road,
                position=lane.position(cop_position, 0),
                speed=self.config["MAX_SPEED"],
            )

            # cop.enable_lane_change = False
            # cop.color = (255, 0, 255)
            self.road.vehicles.append(cop)

    def _remove_and_respawn(self):
        # Remove crashed NPCs (non-cop) and respawn crashed cops behind the ego.
        vehicles_to_remove = []
        cops_to_respawn: List[Cop] = []

        # iterate over a copy since we may remove vehicles from the list
        for vehicle in list(self.road.vehicles):
            if vehicle == self.vehicle:
                continue

            if getattr(vehicle, "crashed", False):
                if isinstance(vehicle, Cop):
                    cops_to_respawn.append(vehicle)
                else:
                    vehicles_to_remove.append(vehicle)

        # Remove crashed NPCs (not cops)
        for vehicle in vehicles_to_remove:
            try:
                self.road.vehicles.remove(vehicle)
            except ValueError:
                # already removed elsewhere
                pass

        # Respawn cops by repositioning them behind the ego and resetting their state
        for cop in cops_to_respawn:
            spawn_dist = self.config.get("cop_respawn_distance", 5.0)
            spawn_speed = self.config["MAX_SPEED"]
            try:
                cop.spawn_behind_ego(distance=spawn_dist, speed=spawn_speed)
            except Exception:
                # If spawn fails for any reason, skip and leave the cop for later
                continue

        return len(vehicles_to_remove)

    def step(self, action):

        if getattr(self, "vehicle", None) is not None:
            # self.vehicle.speed = 0.0
            self.vehicle.position = np.array(self.vehicle.position)

        obs, reward, terminated, truncated, info = super().step(action)

        removed = self._remove_and_respawn()
        if removed > 0:
            # print(f"  Removed {removed} crashed NPC(s)")
            info["npcs_removed"] = removed

        return obs, reward, terminated, truncated, info

    def _reward(self, action):
        # for surviving
        reward = 0.5

        # reward for dodging npcs
        for v in self.copy_of_vehicles:
            if (
                v.speed < 0
                and 2.0 < np.linalg.norm(self.vehicle.position - v.position) < 4.0
            ):
                reward += 0.5
                self.copy_of_vehicles.remove(v)  # no double reward

        # objective: reward for dodging cars as close as possible
        # constraint: avoid getting hit by cops
        for v in self.road.vehicles:
            if v == self.vehicle:
                continue

            # if v.speed > 0:
            #     # reward for getting farther away from cops
            #     dist = np.linalg.norm(self.vehicle.position[0] - v.position[0])
            #     reward += 0.01 * (dist)

            # penalty so no dodging cop
            if v.speed > 0 and v.position[0] > self.vehicle.position[0]:
                reward -= 10

            # penalty for collision with cop
            if v.speed > 0 and self.vehicle.crashed and v.crashed:
                reward += self.config["collision_cop_reward"]

            # penalty for collision with npc
            elif v.speed < 0 and self.vehicle.crashed and v.crashed:
                reward += self.config["collision_reward"]

        # penalty for going off road
        if not self.vehicle.on_road:
            self.vehicle.MAX_SPEED = self.config["MAX_SPEED"] * 0.7
            reward -= 0.3 * abs(self.vehicle.position[1])
        else:
            self.vehicle.MAX_SPEED = self.config["MAX_SPEED"]

        return reward

    def _is_terminated(self):
        return self.vehicle.crashed

    def _is_truncated(self):
        is_time_up = self.time >= self.config["duration"]

        is_oob = self.config["offroad_terminal"] and not self.vehicle.on_road

        return is_time_up or is_oob

    def _info(self, obs, action=None):
        info = super()._info(obs, action)
        info.update(
            {
                "speed": self.vehicle.speed,
                "crashed": self.vehicle.crashed,
                "on_road": self.vehicle.on_road,
            }
        )
        return info


gym.register(
    id="CopChase-v0",
    entry_point="crazy_driver_environment:crazy_driver_env",
)


class Cop(Vehicle):
    def __init__(self, road, position, heading: float = 0.0, speed: float = 0):
        # Vehicle expects (road, position, heading, speed)
        super().__init__(road, position, heading, speed)
        self.enable_lane_change = False
        self.color = (255, 0, 255)

    def spawn_behind_ego(self, distance: float = 5.0, speed=0):
        pos: Vector = self.position
        lane = self.lane

        self.position[0] = self.position[0] - distance

        # Enforce a constant rightward heading for cops (0 radians)
        self.heading = 0.0
        self.speed = speed
        self.action
        self.crashed = False

    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        delta_f = self.action["steering"]
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = self.speed * np.array(
            [np.cos(self.heading + beta), np.sin(self.heading + beta)]
        )
        self.position += v * dt
        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += self.action["acceleration"] * dt
        self.on_state_update()
