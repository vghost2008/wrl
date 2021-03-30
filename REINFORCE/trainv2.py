import gym
from gym.spaces import Discrete,Box
from modeling.mlp import MLP
from REINFORCE.simple_pg import SimplePG
from REINFORCE.simple_pg2 import SimplePG2
from REINFORCE.simple_pg3 import SimplePG3
import tensorflow as tf
import numpy as np
import wnn
from REINFORCE.build import REINFORCE_REGISTRY
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Trainer(object):
    def __init__(self,env_name="CartPole-v0",model='SimplePG3'):
        env = gym.make(env_name)
        assert isinstance(env.observation_space,Box), "This example only works for envs with continuous state spaces."
        assert isinstance(env.action_space,Discrete),"This example only works for envs with discrete action spaces."

        obs_dim = env.observation_space.shape[0]
        n_acts = env.action_space.n

        logits_net = MLP(layers=[32,32,n_acts])
        v_net = MLP(layers=[32,32,1],scope="VNet")
        self.obs = tf.placeholder(dtype=tf.float32,shape=[None,obs_dim])

        self.spg = REINFORCE_REGISTRY.get(model)(net=logits_net,net_v=v_net)

        self.sess = tf.Session()
        self.env = env
        self.render = False
        self.obs = tf.placeholder(dtype=tf.float32,shape=[None,obs_dim])
        self.spg.forward(self.obs)
        self.loss = self.spg.loss()
        self.global_step = tf.train.get_or_create_global_step()
        self.lr = wnn.build_learning_rate(1e-2,
                                          global_step=self.global_step,
                                         lr_decay_type="cosine",
                                         steps=[10000],
                                         decay_factor=0.1,
                                         total_steps=10000,
                                         warmup_steps=200)
        self.opt = wnn.str2optimizer("Momentum", self.lr,momentum=0.9)
        self.train_op, self.total_loss, self.variables_to_train = wnn.nget_train_op(self.global_step, optimizer=self.opt,
                                                                                    clip_norm=32,
                                                                                    loss=self.loss)
        self.batch_size = 5000


    def train_one_epoch(self):
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []
        obs = self.env.reset()
        done = False
        ep_rews = []
        finished_rendering_this_epoch = False
        env = self.env
        env_steps = []

        while True:
            if (not finished_rendering_this_epoch) and self.render:
                env.render()

            batch_obs.append(obs)
            feed_dict = {self.obs:np.reshape(obs,[1,-1])}
            act = self.sess.run(self.spg.action,feed_dict=feed_dict)[0]
            obs,rew,done,_ = env.step(act)
            batch_acts.append(act)
            ep_rews.append(rew)
            if done:
                ep_ret = np.sum(ep_rews)
                batch_rets.append(ep_ret)
                weights = self.spg.get_weights(ep_rews)
                batch_weights += weights
                env_steps.append(len(weights))
                obs,done,ep_rews = env.reset(),False,[]
                finished_rendering_this_epoch = True
                if len(batch_obs)>self.batch_size:
                    break
        base_line = self.sess.run(self.spg.get_base_line(),feed_dict={self.obs:batch_obs})
        feed_dict = {self.obs:batch_obs,
                     self.spg.action_ph:batch_acts,
                     self.spg.weight_ph:np.array(batch_weights)-base_line,
                     self.spg.retgs_ph:batch_weights,
                    }
        loss,global_step,_ = self.sess.run([self.loss,self.global_step,self.train_op],feed_dict=feed_dict)
        print(f"Step {global_step}, loss = {loss}, env steps = {np.mean(env_steps)}")
        return loss,global_step

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        while True:
            loss,global_step = self.train_one_epoch()
            if global_step>10000:
                break

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()

