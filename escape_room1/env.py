"""Environment definition."""

from typing import Tuple, cast

import gym
from gym_sapientino import SapientinoDictSpace
from gym_sapientino.core import configurations
from gym_sapientino.core import actions
from gym_sapientino.core.objects import Robot
from gym_sapientino.core.types import color2int, id2color
from gym_sapientino.wrappers import observations
from gym_sapientino.wrappers.gym import SingleAgentWrapper

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
escape_room1_map_door: Coords = (3, 6)
escape_room1_map_init: Coords = (3, 5)


class DoorDiscreteActions(actions.Command):
    """Discrete actions with modifications .

    This is a GridCommand with one addition: the agent cannot leave the room
    without visiting a special location, the key.
    """

    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    BEEP = 4
    NOP = 5

    def __str__(self) -> str:
        """Get the string representation."""
        if self == self.LEFT:
            return "<"
        elif self == self.RIGHT:
            return ">"
        elif self == self.UP:
            return "^"
        elif self == self.DOWN:
            return "v"
        elif self == self.BEEP:
            return "o"
        elif self == self.NOP:
            return "_"
        else:
            raise ValueError("Shouldn't be here...")

    def _base_step(self, robot: Robot) -> Robot:
        """Move a robot according to the command."""
        x, y = robot.x, robot.y
        if self == self.DOWN:
            y += 1
        elif self == self.UP:
            y -= 1
        elif self == self.RIGHT:
            x += 1
        elif self == self.LEFT:
            x -= 1

        r = Robot(robot.config, x, y, robot.velocity, robot.direction.theta, robot.id)

        return r if not r._on_wall() else robot

    @staticmethod
    def nop() -> "DoorDiscreteActions":
        """Get the NO-OP action."""
        return DoorDiscreteActions.NOP

    @staticmethod
    def beep() -> "DoorDiscreteActions":
        """Get the "Beep" action."""
        return DoorDiscreteActions.BEEP

    def step(self, robot: Robot) -> Robot:
        """Move a robot according to the command."""
        # Update have key
        if (robot.discrete_x, robot.discrete_y) == escape_room1_map_key:
            robot._action_info["has_key"] = True  # type: ignore

        # Try to move
        robot2 = self._base_step(robot)
        robot2._action_info = robot._action_info  # type: ignore

        # Do not move without key
        at_door = ((robot2.discrete_x, robot2.discrete_y) == escape_room1_map_door)
        has_key = robot2._action_info["has_key"]  # type: ignore
        if at_door and not has_key:
            return robot

        return robot2


class LeaveReward(gym.Wrapper):
    """Gerates a single reward when the agent leaves the room.

    Note: assuming the agent is SapientinoDictSpace. 'g' is outside the room.
    """

    def step(self, action):
        """Generates a reward."""
        obs, reward, done, info = super().step(action)
        if obs["color"] == color2int[id2color["g"]]:
            reward += 1
        return obs, reward, done, info


class EscapeRoom1(gym.Wrapper):
    """A spefic gym sapientino environment with temporal goal applied.

    Actions have the usual effect, exept in one location, the door.
    The agent needs to collect the key (a colored cell) before being able to
    traverse the door.
    """

    def __init__(self):
        """Initialize."""
        # Agent configuration
        agent_conf = configurations.SapientinoAgentConfiguration(
            initial_position=escape_room1_map_init,
            commands=DoorDiscreteActions,
        )
        # Env configuration
        env_conf = configurations.SapientinoConfiguration(
            agent_configs=(agent_conf,),
            grid_map=escape_room1_map,
            reward_duplicate_beep=0.0,
            reward_outside_grid=0.0,
            reward_per_step=0.0,
        )
        # Base env
        env = SapientinoDictSpace(configuration=env_conf)

        # Reward
        env = LeaveReward(env)

        # Features
        env = observations.UseFeatures(
            env=cast(SapientinoDictSpace, env),
            features=[observations.DiscreteFeatures],
        )
        env = SingleAgentWrapper(env)

        # Super
        super().__init__(env)
