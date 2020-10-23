import torch


class Optimizer:
    def __init__(self, params_to_optimize, conf,is_adv=False):
        if is_adv:
            self.optim = torch.optim.Adam(params_to_optimize, lr=1e-3, betas=(conf.beta_1, conf.beta_2),eps=conf.epsilon)
        else:
            self.optim = torch.optim.Adam(params_to_optimize, lr=conf.learning_rate, betas=(conf.beta_1, conf.beta_2), eps=conf.epsilon)
        self.scheduler = \
            torch.optim.lr_scheduler.LambdaLR(self.optim,
                                              lr_lambda=lambda epoch: conf.decay ** (float(epoch) / conf.decay_steps))

    def step(self):
        self.optim.step()
        self.scheduler.step()
        #self.optim.zero_grad()


    def zero_grad(self):
        self.optim.zero_grad()

    # @property
    # def lr(self):
    #     return self.scheduler.get_lr()

