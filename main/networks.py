from math import ceil
import sonnet as snt
import tensorflow as tf
from acme.tf import networks
from acme.tf import utils as tf2_utils


class PolicyNetwork(snt.Module):

    def __init__(self, visualRadius, action_size, action_spec):
        super(PolicyNetwork, self).__init__(name="commons-policy")
        self.policy_modules = [
            tf2_utils.batch_concat,
            snt.Conv2D(3, 4, 4, 2, 1),
            tf.nn.relu(),
            snt.Conv2D(3, 8, 4, 2, 1),
            tf.nn.relu(),
            tf.keras.layers.Flatten(),
            snt.Linear(8*((ceil(visualRadius/4))**2), 128),
            tf.nn.relu(),
            snt.Linear(128, action_size),
            networks.TanhToSpec(spec=action_spec)
        ]
        self.model = snt.Sequential(self.policy_modules)

    def __call__(self, x):
        return self.model(x)


class CriticNetwork(snt.Module):

    def __init__(self, visualRadius, action_size, action_spec):
        super(CriticNetwork, self).__init__(name="commons-critic")
        self.model = PolicyNetwork(visualRadius, action_size, action_spec)

    def __call__(self, x):
        return self.model(x)


class ActorNetwork(snt.Module):

    def __init__(self, visualRadius, action_size, action_spec, exploration_sigma):
        super(ActorNetwork, self).__init__(name="commons-actor")
        self.policy_network = PolicyNetwork(
            visualRadius, action_size, action_spec)
        self.behavior_network = self.policy_network + snt.Sequential([networks.ClippedGaussian(exploration_sigma),
                                                                      networks.ClipToSpec(action_spec)])

    def __call__(self, x):
        return self.behavior_network(x)
