# intersection_wrappers.py
#
# Extra safety shaping for intersection-v0:
# - Strongly penalize collisions
# - Penalize driving close behind another car
# - Give small reward for forward progress
# - Penalize sitting still when it would be safe to move

from typing import Tuple, Optional

import numpy as np
import gymnasium as gym


class IntersectionSafetyWrapper(gym.Wrapper):
    """
    Reward shaping for intersection-v0.

    Base env reward still:
      + survival + arrived_reward + high_speed_reward, etc.

    Extra shaping we add:
      - big negative on crash
      - negative when dangerously close to a car in front
      - small positive for forward progress
      - small negative for idling far from the intersection
    """

    def __init__(
        self,
        env: gym.Env,
        crash_penalty: float = -10.0,
        near_dist: float = 10.0,
        near_penalty: float = -0.3,
        very_near_dist: float = 5.0,
        very_near_penalty: float = -0.9,
        progress_scale: float = 0.05,
        idle_penalty: float = -0.02,
        idle_speed_threshold: float = 0.3,
        idle_distance_threshold: float = 12.0,
    ):
        super().__init__(env)

        # Hyper-params you can tweak
        self.crash_penalty = crash_penalty
        self.near_dist = near_dist
        self.near_penalty = near_penalty
        self.very_near_dist = very_near_dist
        self.very_near_penalty = very_near_penalty
        self.progress_scale = progress_scale
        self.idle_penalty = idle_penalty
        self.idle_speed_threshold = idle_speed_threshold
        self.idle_distance_threshold = idle_distance_threshold

        # Track longitudinal progress along the ego's lane
        self._last_s = 0.0

    # -------- helpers --------

    def _get_ego(self):
        base_env = self.env.unwrapped
        ego = getattr(base_env, "vehicle", None)
        if ego is None:
            controlled = getattr(base_env, "controlled_vehicles", [])
            ego = controlled[0] if controlled else None
        return ego, getattr(base_env, "road", None)

    def _longitudinal_position(self, ego) -> float:
        """
        Approximate ego's 's' along its current lane.
        """
        try:
            lane = ego.lane
            s, _ = lane.local_coordinates(ego.position)
            return float(s)
        except Exception:
            # Fallback: just X coordinate
            return float(ego.position[0])

    def _front_vehicle(self, ego, road) -> Tuple[Optional[object], float]:
        """
        Find the closest vehicle *in front* of ego in the same lane-ish direction.
        Returns (veh, approximate_distance_along_lane)
        """
        if road is None:
            return None, float("inf")

        closest_v = None
        min_dist = float("inf")
        ex, ey = ego.position

        for v in road.vehicles:
            if v is ego:
                continue

            vx, vy = v.position
            dx = vx - ex
            dy = vy - ey

            # Only look roughly ahead (assuming vertical lane)
            # For intersection-v0 this simple check is good enough
            if dy <= -1.0:  # car is behind us
                continue

            dist = np.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist
                closest_v = v

        return closest_v, min_dist

    # -------- gym API --------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Initialize reference position
        ego, _ = self._get_ego()
        if ego is not None:
            self._last_s = self._longitudinal_position(ego)
        else:
            self._last_s = 0.0

        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        shaped = 0.0

        ego, road = self._get_ego()
        if ego is None:
            # No extra shaping possible
            return obs, base_reward, terminated, truncated, info

        # 1) Stronger penalty on collision
        crashed = info.get("crashed", False)
        if crashed:
            shaped += self.crash_penalty

        # 2) Penalize being very close to a front vehicle
        front_v, dist = self._front_vehicle(ego, road)
        if front_v is not None and dist < self.near_dist:
            # If we are close AND moving towards it -> high risk
            rel_speed = max(0.0, ego.speed - front_v.speed)

            if dist < self.very_near_dist:
                shaped += self.very_near_penalty * (1.0 + rel_speed / 5.0)
            else:
                shaped += self.near_penalty * (1.0 + rel_speed / 5.0)

        # 3) Reward forward progress towards the intersection/exit
        s = self._longitudinal_position(ego)
        delta_s = max(0.0, s - self._last_s)
        # clip so one crazy fast step doesn't dominate
        delta_s = min(delta_s, 2.0)
        shaped += self.progress_scale * delta_s
        self._last_s = s

        # 4) Penalize idling far away when it's safe (no front car close)
        if (
            ego.speed < self.idle_speed_threshold
            and (front_v is None or dist > self.idle_distance_threshold)
        ):
            shaped += self.idle_penalty

        # Combine
        total_reward = base_reward + shaped

        # log extra for debugging
        info["base_reward"] = base_reward
        info["shaping_reward"] = shaped
        info["total_reward"] = total_reward

        return obs, total_reward, terminated, truncated, info
