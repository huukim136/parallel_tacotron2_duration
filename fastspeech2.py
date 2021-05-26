import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import Encoder, Decoder
from transformer.Layers import PostNet
from modules import TacotronDuration
from utils import get_mask_from_lengths
import hparams as hp
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, use_postnet=True):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder()
        self.variance_adaptor = TacotronDuration()

        self.decoder = Decoder()
        self.mel_linear = nn.Linear(hp.decoder_hidden, hp.n_mel_channels)
        
        self.use_postnet = use_postnet
        if self.use_postnet:
            self.postnet = PostNet()

    def forward(self, src_seq, src_len, mel_len=None, max_src_len=None, max_mel_len=None):
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None
        
        encoder_output, enc_attns = self.encoder(src_seq, src_mask, True)

        variance_adaptor_output, d_prediction, W, pred_mel_mask = self.variance_adaptor(
            encoder_output, src_mask, max_mel_len)
        
        decoder_output, dec_attns, pred_mel_mask = self.decoder(variance_adaptor_output, pred_mel_mask, True)
        mel_output = self.mel_linear(decoder_output)
        
        if self.use_postnet:
            mel_output_postnet = self.postnet(mel_output) + mel_output
        else:
            mel_output_postnet = mel_output

        return mel_output, mel_output_postnet, d_prediction, src_mask, pred_mel_mask, enc_attns, dec_attns, W


if __name__ == "__main__":
    # Test
    model = FastSpeech2(use_postnet=False)
    print(model)
    print(sum(param.numel() for param in model.parameters()))
