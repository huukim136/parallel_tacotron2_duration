import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy
import math

import hparams as hp
import json  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def token_boundary_grid(self, d, mel_length, src_mask, mel_mask):
    	"""
    	PARAMS
    	------
    	d: duration in log scale (batch, text_length)							(B, K)	
    	mel_length: scalar maximum length of mel-spectrogram 					
    	src_mask: text mask (batch, text_length)         						(B, K)
    	mel_mask: text mask (batch, mel_length)          						(B, T)

    	RETURNS
    	-------
    	S and E  
    	"""
        sk = torch.zeros_like(d[:,0].unsqueeze(1))                # (B,1)
        s = torch.zeros_like(d[:,0].unsqueeze(1))              

        for i in range(d.size(1)-1):
            sk = sk + d[:,i].unsqueeze(1)
            s = torch.cat((s, sk), 1)
        e = s + d

        T = torch.linspace(1, mel_length, steps= mel_length).cuda()
        T = torch.log(T.repeat(s.size(0), s.size(1), 1))
        S = T - s.unsqueeze(2).repeat(1,1, T.size(2))
        E = e.unsqueeze(2).repeat(1,1, T.size(2)) - T

        if src_mask is not None:
            E = E.masked_fill(src_mask.unsqueeze(2).expand(-1,-1,E.size(2)), 0.)
            S = S.masked_fill(src_mask.unsqueeze(2).expand(-1,-1,S.size(2)), 0.)
            E = E.masked_fill(mel_mask.unsqueeze(1).expand(-1,E.size(1),-1), 0.)
            S = S.masked_fill(mel_mask.unsqueeze(1).expand(-1,S.size(1),-1), 0.)

        return S, E

    def swish_block(self, S, E, V, mlp1, mlp2, dim):
    	"""
    	PARAMS
    	------
    	S:  (batch, text_length, mel_length)							(B, K, T)						
    	E:  (batch, text_length, mel_length)							(B, K, T)
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

    def forward(self, V, d, mel_length, src_mask, mel_mask):
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
        exp_d = torch.exp(d)

        S, E = self.token_boundary_grid(d, mel_length, src_mask, mel_mask)                        # size (B, K, T)

        # First Swish block + Projection + Softmax
        W = self.swish_block(S, E, V1, self.mlp1_1, self.mlp1_2, self.dim_w)
        W = self.mlp1_3(W)
        W = W.view(S.size(0), S.size(2), S.size(1))                                             # size (B, T, K)
        if src_mask is not None:                                                                # masking in axis d
            W = W.masked_fill(src_mask.unsqueeze(1).expand(-1,W.size(1),-1), -np.inf)
        W  = self.softmax(W)
        if mel_mask is not None:                                                                # masking in axis T
            W = W.masked_fill(mel_mask.unsqueeze(2).expand(-1,-1,W.size(2)), 0.)

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
        if mel_mask is not None:                                                                # masking in axis T
            O = O.masked_fill(mel_mask.unsqueeze(2).expand(-1,-1,256), 0.)
        
        return O, exp_d