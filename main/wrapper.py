"""Wraps the Commons environment to be used as a dm_env environment."""

from typing import List

from acme import specs
from acme import types

import dm_env
import gym
from gym import spaces
import numpy as np

from CommonsGame.envs.env import CommonsGame
from CommonsGame.constants import *

class CommonsWrapper(dm_env.Environment):
  """Environment wrapper for OpenAI Gym environments."""

  def __init__(self, numAgents, visualRadius, mapSketch=bigMap, fullState=False):
    
    self.numAgents = numAgents
    self.visualRadius = visualRadius
    self.mapSketch = bigMap
    self.fullState = fullState
    
    environment = CommonsGame(numAgents, visualRadius, mapSketch=bigMap, fullState=False)
    self._environment = environment
    self._reset_next_step = True

    # Convert action and observation specs.
    obs_space = tuple(self._environment.observation_space for i in range(numAgents))
    act_space = tuple(self._environment.action_space for i in range(numAgents))
    # self._observation_spec = _convert_to_spec(obs_space, name='multi_observation')
    # self._action_spec = _convert_to_spec(act_space, name='multi_action')
    
    self.observation_spec = _convert_to_spec(self._environment.observation_space, name='observation')
    self.action_spec = _convert_to_spec(self._environment.action_space, name='action')
    
    

  def reset(self) -> dm_env.TimeStep:
    """Resets the episode."""
    self._reset_next_step = False
    observation = self._environment.reset()
    observation = [dm_env.restart(obs) for obs in observation]
    return observation

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    """Steps the environment."""
    if self._reset_next_step:
      return self.reset()

    observation, reward, done, info = self._environment.step(list(action))
    transition = []
    for obs,re,d in zip(observation, reward, done) :
      if obs is None or re is None :
        transition.append(None)
      else :
        if d:
          # truncated = info.get('TimeLimit.truncated', False)
          # if truncated:
          #   transition.append(dm_env.truncation(re,obs))
          # else:
          transition.append(dm_env.termination(re,obs))
        else:
          transition.append(dm_env.transition(re,obs))
    return transition

  def observation_spec(self) -> types.NestedSpec:
    return self.observation_spec

  def action_spec(self) -> types.NestedSpec:
    return self.action_spec

  @property
  def environment(self) -> gym.Env:
    """Returns the wrapped environment."""
    return self._environment

  def __getattr__(self, name: str):
    # Expose any other attributes of the underlying environment.
    return getattr(self._environment, name)

  def close(self):
    self._environment.close()


def _convert_to_spec(space: gym.Space, name: str = None) -> types.NestedSpec:
  """Converts an OpenAI Gym space to a dm_env spec or nested structure of specs.
  Box, MultiBinary and MultiDiscrete Gym spaces are converted to BoundedArray
  specs. Discrete OpenAI spaces are converted to DiscreteArray specs. Tuple and
  Dict spaces are recursively converted to tuples and dictionaries of specs.
  Args:
    space: The Gym space to convert.
    name: Optional name to apply to all return spec(s).
  Returns:
    A dm_env spec or nested structure of specs, corresponding to the input
    space.
  """
  if isinstance(space, spaces.Discrete):
    return specs.DiscreteArray(num_values=space.n, dtype=space.dtype, name=name)

  elif isinstance(space, spaces.Box):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=space.low,
        maximum=space.high,
        name=name)

  elif isinstance(space, spaces.MultiBinary):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=0.0,
        maximum=1.0,
        name=name)

  elif isinstance(space, spaces.MultiDiscrete):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=np.zeros(space.shape),
        maximum=space.nvec,
        name=name)

  elif isinstance(space, spaces.Tuple):
    return tuple(_convert_to_spec(s, name) for s in space.spaces)

  elif isinstance(space, spaces.Dict):
    return {
        key: _convert_to_spec(value, name)
        for key, value in space.spaces.items()
    }

  else:
    raise ValueError('Unexpected gym space: {}'.format(space))
