
from typing import Optional, Type

import gym


def find_wrapper(env: gym.Wrapper, wrapper: Type[gym.Wrapper]) -> Optional[gym.Wrapper]:
    """Find a wrapper on the wrapped env hierarchy.

    :return: the instance of wrapper if found, else None.
    """
    if isinstance(env, wrapper):
        return env
    if env.unwrapped is env:
        return None
    return find_wrapper(env.env, wrapper)
