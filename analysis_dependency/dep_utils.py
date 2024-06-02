import torch.nn as nn
import torch


class JSD(nn.Module):
    '''
    Compute Jensen-Shannon divergence for two valid probability distributions.
    '''
    def __init__(self):
        super(JSD, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction='none', log_target=False)

    def forward(self, p, q):
        m = torch.log(0.5 * (p + q)).nan_to_num_()
        kl_p_m = self.kl_loss(m, p).sum(dim=-1, keepdim=True)
        kl_q_m = self.kl_loss(m, q).sum(dim=-1, keepdim=True)
        return 0.5 * (kl_p_m + kl_q_m)
    

def split_sents(sents):
    return sents.split(' ')