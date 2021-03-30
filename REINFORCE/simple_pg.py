import tensorflow as tf
import numpy as np
from .build import REINFORCE_REGISTRY

@REINFORCE_REGISTRY.register()
class SimplePG(object):
    def __init__(self,net):
        self.net = net
        self.action_ph = tf.placeholder(dtype=tf.int32,shape=[None])
        self.weight_ph = tf.placeholder(dtype=tf.float32,shape=[None])
        self.obs = None

    def forward(self,obs):
        self.obs = obs
        self.policy = self.get_policy(obs)
        self.action = self.get_action(self.policy)

    def get_policy(self,obs):
        return tf.distributions.Categorical(self.net.forward(obs))

    def get_action(self,policy):
        return policy.sample()

    def loss(self):
        logp = self.policy.log_prob(self.action_ph)
        return tf.reduce_mean(-logp*self.weight_ph)

    def get_weights(self,ep_rews):
        ep_ret = np.sum(ep_rews)
        weights = [ep_ret]*len(ep_rews)
        return weights
