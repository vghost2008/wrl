from .simple_pg2 import SimplePG2
import numpy as np
import tensorflow as tf
from .build import REINFORCE_REGISTRY

@REINFORCE_REGISTRY.register()
class SimplePG3(SimplePG2):
    def __init__(self,net,net_v):
        super().__init__(net)
        self.net_v = net_v
        self.retgs_ph = tf.placeholder(dtype=tf.float32,shape=[None])

    def forward(self,obs):
        super().forward(obs)
        self.v = tf.squeeze(self.net_v.forward(obs),axis=-1)

    def get_base_line(self):
        return self.v

    def loss(self):
        loss0 = super().loss()
        loss1 = tf.losses.mean_squared_error(labels=self.retgs_ph,predictions=self.v,loss_collection=None)
        loss0 = tf.Print(loss0,["loss",loss0,loss1])
        return loss0+loss1

