from absl.testing import absltest

from acme import wrappers
from acme.testing import fakes

import numpy as np

from wrapper import CommonsWrapper
from CommonsGame.constants import *

class CommonsTest(absltest.TestCase):

  def test_continuous(self):
    env = CommonsWrapper(numAgents=4,visualRadius=2,mapSketch=bigMap, fullState=False)
    self.assertTrue(np.issubdtype(env.observation_spec().dtype, np.uint8))
    self.assertTrue(np.issubdtype(env.action_spec().dtype, np.int64))
    self.assertTrue(np.issubdtype(env.reward_spec().dtype, np.float64))
    self.assertTrue(np.issubdtype(env.discount_spec().dtype, np.float64))

    timestep = env.reset()
    self.assertEqual(timestep.reward, None)
    self.assertEqual(timestep.discount, None)
    self.assertTrue(np.issubdtype(timestep.observation[0].dtype, np.float64))

    timestep = env.step([0.0]*env.numAgents)
    self.assertEqual(len(timestep.reward), env.numAgents)
    self.assertEqual(len(timestep.observation), env.numAgents)
    self.assertTrue(np.issubdtype(timestep.reward[0].dtype, np.int64))
    self.assertTrue(np.issubdtype(type(timestep.discount), np.float64))
    self.assertTrue(np.issubdtype(timestep.observation[0].dtype, np.float64))

  # def test_discrete(self):
  #   env = wrappers.SinglePrecisionWrapper(
  #       fakes.DiscreteEnvironment(
  #           action_dtype=np.int64, obs_dtype=np.int64, reward_dtype=np.float64))

  #   self.assertTrue(np.issubdtype(env.observation_spec().dtype, np.int32))
  #   self.assertTrue(np.issubdtype(env.action_spec().dtype, np.int32))
  #   self.assertTrue(np.issubdtype(env.reward_spec().dtype, np.float32))
  #   self.assertTrue(np.issubdtype(env.discount_spec().dtype, np.float32))

  #   timestep = env.reset()
  #   self.assertEqual(timestep.reward, None)
  #   self.assertEqual(timestep.discount, None)
  #   self.assertTrue(np.issubdtype(timestep.observation.dtype, np.int32))

  #   timestep = env.step(0)
  #   self.assertTrue(np.issubdtype(timestep.reward.dtype, np.float32))
  #   self.assertTrue(np.issubdtype(timestep.discount.dtype, np.float32))
  #   self.assertTrue(np.issubdtype(timestep.observation.dtype, np.int32))


if __name__ == '__main__':
  absltest.main()