import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from transformer.Modules import DepthwiseConv, ConvBlock
import hparams as hp
import pdb


class ConvBranch(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, in_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, in_dim)
        self.n_head = hp.encoder_head
        self.conv_blocks = nn.ModuleList(
            [ConvBlock(in_dim//self.n_head) for _ in range(self.n_head)]
        )
    def forward(self, x):
        # pdb.set_trace()
        x = F.relu(self.linear1(x))
        sz_b, len_q, d_channel = x.size()
        x = x.view(sz_b, len_q, d_channel//self.n_head, self.n_head)
        x = x.unbind(dim=3)

        output = []
        for i in range(self.n_head):
            output.append(self.conv_blocks[i](x[i]))
        # pdb.set_trace()
        output = torch.stack(output)
        
        output = output.view(-1, output.size(2), output.size(3)).permute(1,2,0)
        output = F.relu(self.linear2(output))


        return output