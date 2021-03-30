from .simple_pg import SimplePG
import numpy as np
from .build import REINFORCE_REGISTRY

@REINFORCE_REGISTRY.register()
class SimplePG2(SimplePG):
    @staticmethod
    def reward_to_go(rews):
        retgs = np.zeros([len(rews)+1])
        for i in reversed(range(len(rews))):
            retgs[i] = rews[i]+retgs[i+1]
        return retgs[:-1].tolist()

    def get_weights(self,ep_rews):
        return self.reward_to_go(ep_rews)
