import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL
class GAIL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'GAIL'

    def configure(self, config):
        self.set_common_parameters(config)
        with_global_state = config.sarl.with_global_state

        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))


    def get_attention_weights(self):
        return self.attention_weights