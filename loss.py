import torch
import torch.nn as nn
import hparams as hp
from soft_dtw_cuda import SoftDTW
import torch.nn.functional as F
import pdb
class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self):
        super(FastSpeech2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.sdtw = SoftDTW(use_cuda=True, gamma=0.1)

    def forward(self, d, mel_len, mel, mel_postnet, mel_target, src_mask, pred_mel_mask):
        mel_len.requires_grad = False
        mel_target.requires_grad = False
        d = d.masked_fill(src_mask, 0.)
        ones = torch.ones_like(d).cuda()
        ones = ones.masked_fill(src_mask, 0.)

        mel = mel.masked_fill(pred_mel_mask.unsqueeze(2).repeat(1,1,mel.size(2)), 0.)

        d_loss = torch.mean(torch.abs(mel_len - torch.sum(d, dim=1))/torch.sum(ones, dim =1))
        mel_loss = 0.001*torch.mean(self.sdtw(F.sigmoid(mel), F.sigmoid(mel_target)))
        mel_postnet_loss = 0.001*torch.mean(self.sdtw(F.sigmoid(mel_postnet), F.sigmoid(mel_target)))

        
        return mel_loss, mel_postnet_loss, d_loss
