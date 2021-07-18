
import time
from typing import Any, Tuple, cast

import gym
from gym.spaces import Discrete

from escape_room1.env import EscapeRoom1

Action = int
GymStep = Tuple[Any, float, bool, dict]


def test(env: gym.Env):
    """Test loop."""
    # Episodes
    for _ in range(10):

        # Init episode
        obs = env.reset()
        reward = None
        done = False
        info = None

        while not done:
            # Render
            env.render()

            # Maybe interact
            action = _interact((obs, reward, done, info), env=env)
            if action < 0:
                break

            # Move env
            obs, reward, done, info = env.step(action)

            if done:
                print(f"Env reset: obs {obs}, reward {reward}")

            # Let us see the screen
            time.sleep(0.1)


def _interact(data: GymStep, env: gym.Env) -> Action:
    """Interact with user.

    The function shows some data, then asks for an action on the command
    line.
    :param stata: the last tuple returned by gym environment.
    :return: the action to perform.
    """
    print("Env step")
    print("  Observation:", data[0])
    print("       Reward:", data[1])
    print("         Done:", data[2])
    print("        Infos:", data[3])

    action_space = cast(Discrete, env.action_space)
    act = input("Action in [-1, {}]: ".format(action_space.n - 1))
    action = int(act)
    if action < 0:
        print("Reset")

    return action


env = EscapeRoom1()
test(env)
