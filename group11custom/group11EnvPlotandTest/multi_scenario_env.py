# multi_scenario_env.py

import random
from typing import List, Optional, Tuple, Dict, Any

import gymnasium as gym
import highway_env  # noqa: F401  # needed to register highway-env environments
from gymnasium import spaces

from highway_env.vehicle.behavior import AggressiveVehicle

from intersection_helpers import IntersectionSafetyWrapper


class MultiScenarioHighwayEnv(gym.Env):
    """
    Custom environment that randomly switches between:
      - highway-v0
      - merge-v0
      - intersection-v0

    Each episode:
      * picks one scenario at random,
      * configures traffic based on a shared 'aggressiveness' parameter,
      * uses a common observation config (e.g. Lidar),
      * exposes a single observation_space/action_space to the agent.

    The 'aggressiveness' parameter (in [0, 1]) can be used for curriculum:
      - 0.0 → easy traffic
      - 1.0 → dense / antagonistic traffic
    """

    metadata = {"render_modes": ["rgb_array", "human", None]}

    def __init__(
        self,
        env_ids: Optional[List[str]] = None,
        observation_config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        aggressiveness: float = 0.0,
    ):
        super().__init__()

        # Which base scenarios to include
        if env_ids is None:
            env_ids = ["highway-v0", "merge-v0", "intersection-v0"]
        self.env_ids = env_ids
        self.render_mode = render_mode
        self._seed = seed

        # Curriculum / difficulty knob: 0.0 = easy, 1.0 = hardest
        self.aggressiveness: float = float(max(0.0, min(1.0, aggressiveness)))

        # Default observation: LidarObservation shared across all scenarios
        if observation_config is None:
            observation_config = {
                "type": "LidarObservation",
                "cells": 32,
                "maximum_range": 60,
                "normalize": True,
            }
        self.observation_config = observation_config

        # Create one instance of each underlying env upfront
        self._envs: Dict[str, gym.Env] = {}
        for eid in self.env_ids:
            base_config = {
                "observation": self.observation_config,
                "simulation_frequency": 15,
                "policy_frequency": 5,
                # We'll override vehicles_count etc. *per reset* based on aggressiveness
            }

            env = gym.make(eid, config=base_config, render_mode=self.render_mode)
            if seed is not None:
                env.reset(seed=seed)

            # Only intersection-v0 gets the safety/reward shaping wrapper
            if eid == "intersection-v0":
                env = IntersectionSafetyWrapper(env)

            self._envs[eid] = env

        # Assume all scenarios share the same obs/action spaces (true for default configs)
        first_env = self._envs[self.env_ids[0]]
        self.observation_space: spaces.Space = first_env.observation_space
        self.action_space: spaces.Space = first_env.action_space

        self.current_env_id: Optional[str] = None
        self.current_env: Optional[gym.Env] = None

    # ---------- Curriculum / aggressiveness control ----------

    def set_curriculum_progress(self, progress: float) -> None:
        """
        Set aggressiveness based on curriculum progress in [0, 1].
        You can call this from a PPO callback during training.

        Example schedule:
            progress = num_timesteps / total_timesteps
        """
        self.aggressiveness = float(max(0.0, min(1.0, progress)))

    # ---------- Core env API ----------

    def _select_scenario(self) -> None:
        """Randomly choose one of the configured scenarios for this episode."""
        self.current_env_id = random.choice(self.env_ids)
        self.current_env = self._envs[self.current_env_id]

    def _build_scenario_config(self, env_id: str) -> Dict[str, Any]:
        """
        Build a per-scenario config that depends on self.aggressiveness.
        You can tweak these formulas to make traffic more 'antagonistic'.

        IMPORTANT:
        We also force the same 5-action DiscreteMetaAction for ALL scenarios,
        so PPO always sees a consistent action space.
        """
        # Base vehicles per scenario
        base_vehicles = {
            "highway-v0": 25,
            "merge-v0": 22,
            "intersection-v0": 20,
        }
        base = base_vehicles.get(env_id, 20)

        # Increase traffic with aggressiveness (0.0 → base, 1.0 → ~2.5x base)
        vehicles_count = int(base * (1.0 + 1.5 * self.aggressiveness))

        # Shared config: same observation + SAME ACTION SPACE for all envs
        cfg: Dict[str, Any] = {
            "observation": self.observation_config,
            "vehicles_count": vehicles_count,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "action": {
                # This gives us the 5-action ACTIONS_ALL:
                # 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True,
            },
        }

        # Example: tweak duration per scenario
        if env_id == "highway-v0":
            cfg["duration"] = 35  # seconds
        elif env_id == "merge-v0":
            cfg["duration"] = 35
        elif env_id == "intersection-v0":
            cfg["duration"] = 100

        return cfg

    def reset(self, seed=None, options=None):
        self._select_scenario()
        assert self.current_env is not None
        env_id = self.current_env_id

        # Configure based on current aggressiveness *before* reset
        cfg = self._build_scenario_config(env_id)
        try:
            self.current_env.unwrapped.configure(cfg)
        except AttributeError:
            # If for some reason configure doesn't exist, just skip
            pass

        # Now reset (optionally reseed)
        if seed is not None:
            obs, info = self.current_env.reset(seed=seed, options=options)
        else:
            obs, info = self.current_env.reset(options=options)

        # Make some background cars antagonistic
        self._inject_antagonistic_vehicles()

        return obs, info

    def _inject_antagonistic_vehicles(self):
        """
        Turn a fraction of background cars into 'antagonistic' AggressiveVehicles.

        - Fraction scales with self.aggressiveness (0..1)
        - Max fraction is 50% of non-ego vehicles
        - These AggressiveVehicles are drawn yellow and have more assertive behaviour
        """
        if self.current_env is None or self.aggressiveness <= 0.0:
            return

        # 0.0 → 0%, 1.0 → 30% antagonistic
        max_fraction = 0.30
        frac = max_fraction * float(self.aggressiveness)

        base_env = self.current_env.unwrapped
        road = base_env.road
        ego = base_env.vehicle

        # All non-ego vehicles currently on the road
        others = [v for v in road.vehicles if v is not ego]
        n = len(others)
        if n == 0 or frac <= 0.0:
            return

        n_ant = max(1, int(n * frac))

        # Pick the cars closest to the ego vehicle to make them antagonistic
        def dist_to_ego(v):
            try:
                return abs(v.lane_distance_to(ego))
            except Exception:
                return float("inf")

        others_sorted = sorted(others, key=dist_to_ego)
        antagonists = set(others_sorted[:n_ant])

        new_list = []
        for v in road.vehicles:
            if v is ego:
                new_list.append(v)
                continue

            if v in antagonists:
                # Replace with a more aggressive behaviour vehicle,
                # keeping approximately the same state.
                target_lane = getattr(
                    v, "target_lane_index", getattr(v, "lane_index", None)
                )
                target_speed = getattr(v, "target_speed", v.speed * 1.2)

                ag = AggressiveVehicle(
                    road=road,
                    position=v.position,
                    heading=v.heading,
                    speed=v.speed,
                    target_lane_index=target_lane,
                    target_speed=target_speed,
                    # route=getattr(v, "route", None),
                    enable_lane_change=True,
                    timer=getattr(v, "timer", None),
                )

                # mark as antagonist so we can steer it towards the ego later
                ag.is_antagonist = True

                # push them a bit "towards" the ego by increasing target_speed
                ag.target_speed = max(
                    getattr(ego, "target_speed", getattr(ego, "speed", 0.0) + 5.0),
                    ag.target_speed,
                )

                new_list.append(ag)
            else:
                new_list.append(v)

        road.vehicles = new_list

    def _update_antagonists_targets(self) -> None:
        """
        On each step, adjust antagonistic vehicles so they more directly
        'target' the ego:
          - aggressively try to get into the ego's lane
          - go noticeably faster than the ego, especially when behind
        """
        if self.current_env is None:
            return

        base_env = self.current_env.unwrapped
        road = getattr(base_env, "road", None)
        if road is None:
            return

        # Find ego vehicle
        ego = getattr(base_env, "vehicle", None)
        if ego is None:
            controlled = getattr(base_env, "controlled_vehicles", [])
            ego = controlled[0] if controlled else None
        if ego is None:
            return

        ego_lane = getattr(ego, "lane_index", None)
        ego_speed = getattr(ego, "speed", 0.0)
        ex, ey = getattr(ego, "position", (0.0, 0.0))

        for v in getattr(road, "vehicles", []):
            if not getattr(v, "is_antagonist", False):
                continue

            vx, vy = getattr(v, "position", (0.0, 0.0))

            # Rough "ahead/behind" check along x-axis
            longitudinal_delta = ex - vx  # >0 means v is behind ego, <0 means ahead

            # 1) Try to get into the SAME lane as the ego (direct targeting)
            if ego_lane is not None and hasattr(v, "target_lane_index"):
                try:
                    v.target_lane_index = ego_lane
                except Exception:
                    pass

            # 2) Chase the ego with higher speed
            if hasattr(v, "target_speed"):
                base_offset = 6.0  # a bit more than ego

                if longitudinal_delta > 20.0:  # far behind
                    chase_offset = base_offset + 6.0  # ego + 12
                elif longitudinal_delta > 5.0:  # somewhat behind
                    chase_offset = base_offset + 3.0  # ego + 9
                else:
                    chase_offset = base_offset  # ego + 6

                desired = ego_speed + chase_offset

                try:
                    v.target_speed = max(desired, getattr(v, "target_speed", 0.0))
                except Exception:
                    v.target_speed = desired

    def step(self, action):
        """Step through the current selected scenario."""
        assert self.current_env is not None, "Call reset() before step()."

        self._update_antagonists_targets()

        obs, reward, terminated, truncated, info = self.current_env.step(action)
        info = info or {}
        info["scenario"] = self.current_env_id
        info["aggressiveness"] = self.aggressiveness
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.current_env is None:
            return None
        return self.current_env.render()

    def close(self):
        for env in self._envs.values():
            env.close()
