import torch

class TransformerOptimizer:
    def __init__(self, optimizer, init_lr=2, warmup_steps=4000):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.warmup_step = 4000
        self.step_num = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.step_num += 1
        lr = self.init_lr * min(self.step_num**(-0.5),
                                self.step_num*(self.warmup_step**(-1.5)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.optimizer.step()


    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()
