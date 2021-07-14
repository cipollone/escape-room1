"""Environment definition."""

from typing import Tuple, Dict, Any

import gym
from gym_sapientino import SapientinoDictSpace
from gym_sapientino.core import configurations
from gym_sapientino.core import actions
from gym_sapientino.core.objects import Robot

Coords = Tuple[int, int]


escape_room1_map = """\
|#######    |
|#r    #    |
|#     #    |
|#        g |
|#     #    |
|#     #    |
|#######    |"""
escape_room1_map_key: Coords = (1, 1)
escape_room1_map_door: Coords = (4, 7)


class DoorDiscreteActions(actions.GridCommand):
    """Discrete actions.

    Discrete actions with one exeption: you cannot leave the room if you
    don't visit 'r' first.
    """

    def reset(self, robot: Robot):
        """Just a callback."""
        robot._action_info = dict(has_key=False)  # type: ignore

    def step(self, robot: Robot) -> Robot:
        """Move a robot according to the command."""
        # Update have key
        if (robot.discrete_x, robot.discrete_y) == escape_room1_map_key:
            robot._action_info["has_key"] = True  # type: ignore

        # Try to move
        robot2 = super().step(robot)
        robot2._action_info = robot._action_info  # type: ignore

        # Do not move without key
        at_door = ((robot2.discrete_x, robot2.discrete_y) == escape_room1_map_door)
        has_key = robot2._action_info["has_key"]  # type: ignore
        if at_door and not has_key:
            return robot

        return robot2


class EscapeRoom1(gym.Wrapper):
    """A spefic gym sapientino environment with temporal goal applied.

    Actions have the usual effect, exept in one location, the door.
    We need to be passed from the key in order for the door to open.
    """
    metadata = {
        "render.modes": ["human"],
    }
    reward_range = (0.0, 1.0)

    def __init__(self):
        """Initialize."""

        # Set action space

        # Set observation space
