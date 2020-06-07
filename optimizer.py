import torch
from pdb import set_trace as bp

class TransformerOptimizer:
    def __init__(self, optimizer, scale_factor, warmup_step, d_model=512):
        self.optimizer = optimizer
        self.k = scale_factor
        self.init_lr = d_model ** (-0.5)
        self.warmup_step = warmup_step
        self.step_num = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.step_num += 1
        lr = self.k * self.init_lr * min(self.step_num**(-0.5),
                                         self.step_num*(self.warmup_step**(-1.5)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.optimizer.step()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.k = state_dict['k']
        print(self.k, flush=True)
        self.init_lr = state_dict['init_lr']
        self.warmup_step = state_dict['warmup_step']
        self.step_num = state_dict['step_num']


    def state_dict(self):
        state_dict = {'optimizer': self.optimizer.state_dict(),
                      'k': self.k,
                      'init_lr': self.init_lr,
                      'warmup_step': self.warmup_step,
                      'step_num': self.step_num}
        return state_dict
