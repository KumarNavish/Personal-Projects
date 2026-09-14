# Deployment adapter for La Javaness L2RPN 2023 topology actor.
# Original API: MPL-2.0, Copyright 2023-2024 La Javaness. See bundled AUTHORS and LICENSE.
# This adapter changes the inference implementation, not the weights or action list.
import json
from pathlib import Path
import numpy as np
from .base_module import GreedyModule

class TopoNNTopKModule(GreedyModule):
    def __init__(self, action_space, gym_env, model_path, top_k=1, device='cpu'):
        super().__init__(action_space)
        self.gym_env=gym_env;self.top_k=top_k;self.device='cpu'
        root=Path(__file__).resolve().parents[1]/'models'
        self.layers=json.loads((root/'actor.json').read_text())
        self.weights=dict(np.load(root/'actor.npz',allow_pickle=False))
    def logits(self,gym_obs):
        x=np.asarray(gym_obs,dtype=np.float32)
        for layer in self.layers:
            op=layer['op']
            if op=='linear':x=x@self.weights[layer['w']].T+self.weights[layer['b']]
            elif op=='tanh':x=np.tanh(x)
            elif op=='relu':x=np.maximum(x,0)
            elif op=='elu':x=np.where(x>=0,x,np.expm1(x))
            else:raise ValueError('Unsupported actor operation')
        if not np.isfinite(x).all():raise ValueError('Nonfinite actor output')
        return x
    def get_top_k(self,gym_obs,top_k):
        return np.argsort(-self.logits(gym_obs),kind='stable')[:top_k]
    def _get_tested_action(self,observation):
        x=self.gym_env.observation_space.to_gym(observation)
        return [self.gym_env.action_space.from_gym(i) for i in self.get_top_k(x,self.top_k)]
