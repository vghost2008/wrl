import tensorflow as tf

slim = tf.contrib.slim

class MLP(object):
    def __init__(self,layers=[64,64],scope="MLP"):
        self.layers = layers
        self.scope = scope

    def forward(self,net):
        with tf.variable_scope(self.scope):
            net = tf.expand_dims(net,axis=0)
            for i,dim in enumerate(self.layers):
                net = slim.fully_connected(net,dim,activation_fn=tf.nn.relu if i != len(self.layers)-1 else None)

            net = tf.squeeze(net,axis=0)

            return net
