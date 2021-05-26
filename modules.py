import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import numpy as np
import copy
import math

from torchsearchsorted import searchsorted
from utils import get_mask_from_lengths

import hparams as hp
import utils
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TacotronDuration(nn.Module):
    """ Variance Adaptor """

    def __init__(self):
        super(TacotronDuration, self).__init__()
        self.duration_predictor = DurationPredictor()
        self.learned_upsample = LearnedUpsample()
        
        self.pitch_bins = nn.Parameter(torch.exp(torch.linspace(np.log(hp.f0_min), np.log(hp.f0_max), hp.n_bins-1)))
        self.energy_bins = nn.Parameter(torch.linspace(hp.energy_min, hp.energy_max, hp.n_bins-1))
        self.pitch_embedding = nn.Embedding(hp.n_bins, hp.encoder_hidden)
        self.energy_embedding = nn.Embedding(hp.n_bins, hp.encoder_hidden)
    
    def forward(self, x, src_mask, mel_len=None, max_len=None):

        V, duration_prediction = self.duration_predictor(x, src_mask)
        x, W, pred_mel_mask = self.learned_upsample(V, duration_prediction, src_mask)

        return x, duration_prediction, W, pred_mel_mask


class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self):
        super(VariancePredictor, self).__init__()

        self.input_size = hp.encoder_hidden
        self.filter_size = hp.variance_predictor_filter_size
        self.kernel = hp.variance_predictor_kernel_size
        self.conv_output_size = hp.variance_predictor_filter_size
        self.dropout = hp.variance_predictor_dropout

        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv1d_1", Conv(self.input_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=(self.kernel-1)//2)),
            ("relu_1", nn.ReLU()),
            ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            ("dropout_1", nn.Dropout(self.dropout)),
            ("conv1d_2", Conv(self.filter_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("relu_2", nn.ReLU()),
            ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            ("dropout_2", nn.Dropout(self.dropout))
        ]))

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)
        
        if mask is not None:
            out = out.masked_fill(mask, 0.)
        
        return out

class DurationPredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self):
        super(DurationPredictor, self).__init__()

        self.input_size = hp.encoder_hidden
        self.filter_size = hp.filter_size
        self.kernel = hp.kernel
        self.conv_output_size = hp.filter_size
        self.dropout = hp.dropout

        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv1d_1", Conv(self.input_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=(self.kernel-1)//2)),
            ("relu_1", nn.ReLU()),
            ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            ("dropout_1", nn.Dropout(self.dropout)),
            ("conv1d_2", Conv(self.filter_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("relu_2", nn.ReLU()),
            ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            ("dropout_2", nn.Dropout(self.dropout))
        ]))

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, encoder_output, mask):
        """
        PARAMS
        ------
        encoder_output: output of the text encoder (batch, text_length, hidden_dim)
        mask: text mask (batch, text_length)

        RETURNS
        -------
        representations V 
        duration in log scale d
        """
        V = self.conv_layer(encoder_output)
        d = self.softplus(self.linear_layer(V))
        d = d.squeeze(-1)
        
        if mask is not None:
            d = d.masked_fill(mask, 0.)
        
        return V, d

class LearnedUpsample(nn.Module):

    def __init__(self):
        super(LearnedUpsample, self).__init__()

        self.input_size = hp.encoder_hidden
        self.kernel = hp.upsample_conv1d_kernel                                                                         
        self.upsample_conv1d_dim_w = hp.upsample_conv1d_dim_w
        self.upsample_conv1d_dim_c = hp.upsample_conv1d_dim_c

        self.conv1D_1 = nn.Sequential(OrderedDict([
            ("conv1d", Conv(self.input_size,
                              self.upsample_conv1d_dim_w,
                              kernel_size=self.kernel,
                              padding=(self.kernel-1)//2)),
            ("layer_norm", nn.LayerNorm(self.upsample_conv1d_dim_w))
        ]))

        self.conv1D_2 = nn.Sequential(OrderedDict([
            ("conv1d", Conv(self.input_size,
                              self.upsample_conv1d_dim_c,
                              kernel_size=self.kernel,
                              padding=(self.kernel-1)//2)),
            ("layer_norm", nn.LayerNorm(self.upsample_conv1d_dim_c))
        ]))

        # First Swish block + Projection + Softmax
        self.mlp1_1 = nn.Linear(hp.mlp_dim_w, hp.mlp_dim_w)
        self.mlp1_2 = nn.Linear(hp.mlp_dim_w, hp.mlp_dim_w)
        self.mlp1_3 = nn.Linear(hp.mlp_dim_w, 1) 
        self.softmax = nn.Softmax(dim = 2)
        self.dim_w = hp.mlp_dim_w

        # Second Swish block for measuring C
        self.dim_c = hp.mlp_dim_c//2    
        self.mlp2_1 = nn.Linear(self.dim_c, self.dim_c)
        self.mlp2_2 = nn.Linear(self.dim_c, self.dim_c)
        self.mlp2_3 = nn.Linear(self.dim_c, self.dim_c)
        self.mlp2_4 = nn.Linear(self.dim_c, self.dim_c)
        self.A = nn.Parameter(torch.normal(mean = 0., std = 1.0, size=(hp.mlp_dim_c, hp.encoder_hidden)).cuda())

    def token_boundary_grid(self, d, src_mask):
        """
        PARAMS
        ------
        d: duration in log scale (batch, text_length)                           (B, K)  
        mel_length: scalar maximum length of mel-spectrogram                    
        src_mask: text mask (batch, text_length)                                (B, K)

        RETURNS
        -------
        S and E  
        """
        # pdb.set_trace()
        sk = torch.zeros_like(d[:,0].unsqueeze(1))                # (B,1)
        s = torch.zeros_like(d[:,0].unsqueeze(1))              

        for i in range(d.size(1)-1):
            sk = sk + d[:,i].unsqueeze(1)
            s = torch.cat((s, sk), 1)
        e = s + d

        T = torch.round(torch.sum(d, dim=-1, keepdim=True)) #[B, 1] 
        mel_mask = get_mask_from_lengths(T.squeeze(1), int(torch.max(T).item()))
        T = torch.range(1, torch.max(T).item()).to(device)
        T = T.repeat(s.size(0), s.size(1), 1)
        S = T - s.unsqueeze(2).repeat(1,1, T.size(2))
        E = e.unsqueeze(2).repeat(1,1, T.size(2)) - T

        if src_mask is not None:
            E = E.masked_fill(src_mask.unsqueeze(2).expand(-1,-1,E.size(2)), 0.)
            S = S.masked_fill(src_mask.unsqueeze(2).expand(-1,-1,S.size(2)), 0.)
            E = E.masked_fill(mel_mask.unsqueeze(1).expand(-1,E.size(1),-1), 0.)
            S = S.masked_fill(mel_mask.unsqueeze(1).expand(-1,S.size(1),-1), 0.)

        return S, E, mel_mask

    def swish_block(self, S, E, V, mlp1, mlp2, dim):
        """
        PARAMS
        ------
        S:  (batch, text_length, mel_length)                            (B, K, T)                       
        E:  (batch, text_length, mel_length)                            (B, K, T)
        V: output of conv block (batch, text_length, hidden_dim)        (B, K, H)
        dim : dimension

        RETURNS
        -------
        out 
        """
        S = S.unsqueeze(3).repeat(1,1,1,dim)
        E = E.unsqueeze(3).repeat(1,1,1,dim)
        V = V.unsqueeze(2).repeat(1,1,E.size(2),1)

        V = S + E + V
        out = mlp1(V.view(-1, V.size(3)))
        out = out * F.sigmoid(out)                                                                    # Swish activation
        out = mlp2(out)
        out = out * F.sigmoid(out)    
        return out                                                                                    # Swish activation

    def forward(self, V, d, src_mask):
        """
        PARAMS
        ------
        V: (batch, text_length, hidden_dim)              (B, K, H)
        d: duration in log scale (batch, text_length)    (B, K)
        mel_length: scalar maximum length of mel-spectrogram        
        src_mask: text mask (batch, text_length)         (B, K)
        mel_mask: text mask (batch, mel_length)          (B, T)

        RETURNS
        -------
        O 
        duration in normal scale exp_d
        """
        V1 = self.conv1D_1(V)
        V1 = V1 * F.sigmoid(V1)
        V2 = self.conv1D_2(V)
        V2 = V2 * F.sigmoid(V2)

        S, E, pred_mel_mask = self.token_boundary_grid(d, src_mask)                        # size (B, K, T)

        # First Swish block + Projection + Softmax
        W = self.swish_block(S, E, V1, self.mlp1_1, self.mlp1_2, self.dim_w)
        W = self.mlp1_3(W)
        W = W.view(S.size(0), S.size(2), S.size(1))                                         # size (B, T, K)
                                                                       
        W = W.masked_fill(src_mask.unsqueeze(1).expand(-1,W.size(1),-1), -np.inf)           # masking in axis d
        W  = self.softmax(W)
        W = W.masked_fill(pred_mel_mask.unsqueeze(2).expand(-1,-1,W.size(2)), 0.)                # masking in axis T

        # Second Swish block for measuring C
        C1 = self.swish_block(S, E, V2, self.mlp2_1, self.mlp2_2, self.dim_c)
        C1 = C1.view(S.size(0), S.size(2), S.size(1), self.dim_c)                                             # size (B, T, K)
        C2 = self.swish_block(S, E, V2, self.mlp2_3, self.mlp2_3, self.dim_c)
        C2 = C2.view(S.size(0), S.size(2), S.size(1), self.dim_c)                                             # size (B, T, K)
        C = torch.cat((C1,C2), dim=3)

        # As in the paper: O  = left term + right term
        left_term = torch.bmm(W, V)
        right_term = torch.bmm(torch.einsum('btk,btkp->btp', W, C), self.A.unsqueeze(0).expand(W.size(0), -1, -1))

        O = left_term + right_term
        if pred_mel_mask is not None:                                                                # masking in axis T
            O = O.masked_fill(pred_mel_mask.unsqueeze(2).expand(-1,-1,256), 0.)
        
        return O, W, pred_mel_mask

class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
