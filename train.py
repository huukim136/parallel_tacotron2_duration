import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboar import SummaryWriter
from tensorboardX import SummaryWriter

import numpy as np
import argparse
import os
import time

from fastspeech2 import FastSpeech2
from loss import FastSpeech2Loss
from dataset import Dataset
from optimizer import ScheduledOptim
from evaluate import evaluate
import hparams as hp
import utils
import audio as Audio
import pdb
import random
from utils import plot_alignment_to_numpy


def plot_attn(train_logger, enc_attns, dec_attns, current_step, hp):
    # pdb.set_trace()

    idx = random.randint(0, enc_attns[0].size(0) - 1)
    for i in range(len(enc_attns)):
        train_logger.add_image(
            "encoder.attns_layer_%s"%i,
            plot_alignment_to_numpy(enc_attns[i].data.cpu().numpy()[idx].T),
            current_step)

        train_logger.add_image(
            "decoder.attns_layer_%s"%i,
            plot_alignment_to_numpy(dec_attns[i].data.cpu().numpy()[idx].T),
            current_step)
     

def main(args):
    torch.manual_seed(0)

    # Get device
    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')
    
    # Get dataset
    dataset = Dataset("train.txt") 
    loader = DataLoader(dataset, batch_size=hp.batch_size**2, shuffle=True, 
        collate_fn=dataset.collate_fn, drop_last=True, num_workers=0)

    # Define model
    # model = nn.DataParallel(FastSpeech2()).to(device)
    model = FastSpeech2().to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of FastSpeech2 Parameters:', num_param)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), betas=hp.betas, eps=hp.eps, weight_decay = hp.weight_decay)
    scheduled_optim = ScheduledOptim(optimizer, hp.decoder_hidden, hp.n_warm_up_step, args.restore_step)
    Loss = FastSpeech2Loss().to(device) 
    print("Optimizer and Loss Function Defined.")

    # Load checkpoint if exists
    checkpoint_path = os.path.join(hp.checkpoint_path)
    try:
        checkpoint = torch.load(os.path.join(
            checkpoint_path, 'checkpoint_{}.pth.tar'.format(args.restore_step)))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step {}---\n".format(args.restore_step))
    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    # Load vocoder
    if hp.vocoder == 'melgan':
        melgan = utils.get_melgan()
        melgan.to(device)
    elif hp.vocoder == 'waveglow':
        waveglow = utils.get_waveglow()
        waveglow.to(device)

    # Init logger
    log_path = hp.log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(os.path.join(log_path, 'train'))
        os.makedirs(os.path.join(log_path, 'validation'))
    train_logger = SummaryWriter(os.path.join(log_path, 'train'))
    val_logger = SummaryWriter(os.path.join(log_path, 'validation'))

    # Init synthesis directory
    synth_path = hp.synth_path
    if not os.path.exists(synth_path):
        os.makedirs(synth_path)

    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()
    
    # Training
    model = model.train()
    # pdb.set_trace()
    for epoch in range(hp.epochs):
        # Get Training Loader
        total_step = hp.epochs * len(loader) * hp.batch_size

        for i, batchs in enumerate(loader):
            for j, data_of_batch in enumerate(batchs):
                start_time = time.perf_counter()

                current_step = i*hp.batch_size + j + args.restore_step + epoch*len(loader)*hp.batch_size + 1
                
                # Get Data
                text = torch.from_numpy(data_of_batch["text"]).long().to(device)
                mel_target = torch.from_numpy(data_of_batch["mel_target"]).float().to(device)
                D = torch.from_numpy(data_of_batch["D"]).long().to(device)
                log_D = torch.from_numpy(data_of_batch["log_D"]).float().to(device)
                f0 = torch.from_numpy(data_of_batch["f0"]).float().to(device)
                energy = torch.from_numpy(data_of_batch["energy"]).float().to(device)
                src_len = torch.from_numpy(data_of_batch["src_len"]).long().to(device)
                mel_len = torch.from_numpy(data_of_batch["mel_len"]).long().to(device)
                max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
                max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)
                
                # Forward
                mel_output, mel_postnet_output, duration_output, src_mask, pred_mel_mask, enc_attns, dec_attns, W = model(
                    text, src_len, mel_len, max_src_len, max_mel_len)
                # Cal Loss
                mel_loss, mel_postnet_loss, d_loss= Loss(
                        duration_output, mel_len, mel_output, mel_postnet_output, mel_target, src_mask, pred_mel_mask)
                total_loss = mel_loss + mel_postnet_loss + d_loss
                 
                # Logger
                t_l = total_loss.item()
                m_l = mel_loss.item()
                m_p_l = mel_postnet_loss.item()
                d_l = d_loss.item()
                 
                # Backward
                total_loss = total_loss / hp.acc_steps
                total_loss.backward()
                if current_step % hp.acc_steps != 0:
                    continue

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)

                # Update weights
                scheduled_optim.step_and_update_lr()
                scheduled_optim.zero_grad()
                
                # Print
                if current_step % hp.log_step == 0:
                    Now = time.perf_counter()

                    str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                        epoch+1, hp.epochs, current_step, total_step)
                    str2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f};".format(
                        t_l, m_l, m_p_l, d_l)
                    str3 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                        (Now-Start), (total_step-current_step)*np.mean(Time))

                    print("\n" + str1)
                    print(str2)
                    print(str3)
                    
                    with open(os.path.join(log_path, "log.txt"), "a") as f_log:
                        f_log.write(str1 + "\n")
                        f_log.write(str2 + "\n")
                        f_log.write(str3 + "\n")
                        f_log.write("\n")

                    train_logger.add_scalar('Loss/total_loss', t_l, current_step)
                    train_logger.add_scalar('Loss/mel_loss', m_l, current_step)
                    train_logger.add_scalar('Loss/mel_postnet_loss', m_p_l, current_step)
                    train_logger.add_scalar('Loss/duration_loss', d_l, current_step)

                    plot_attn(train_logger, enc_attns, dec_attns, current_step, hp)
                
                if current_step % hp.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(checkpoint_path, 'checkpoint_{}.pth.tar'.format(current_step)))
                    print("save model at step {} ...".format(current_step))

                if current_step % hp.synth_step == 0:
                    length = mel_len[0].item()
                    mel_target_torch = mel_target[0, :length].detach().unsqueeze(0).transpose(1, 2)
                    mel_target = mel_target[0, :length].detach().cpu().transpose(0, 1)
                    mel_torch = mel_output[0, :length].detach().unsqueeze(0).transpose(1, 2)
                    mel = mel_output[0, :length].detach().cpu().transpose(0, 1)
                    mel_postnet_torch = mel_postnet_output[0, :length].detach().unsqueeze(0).transpose(1, 2)
                    mel_postnet = mel_postnet_output[0, :length].detach().cpu().transpose(0, 1)
                    Audio.tools.inv_mel_spec(mel, os.path.join(synth_path, "step_{}_griffin_lim.wav".format(current_step)))
                    Audio.tools.inv_mel_spec(mel_postnet, os.path.join(synth_path, "step_{}_postnet_griffin_lim.wav".format(current_step)))
                    
                    if hp.vocoder == 'melgan':
                        utils.melgan_infer(mel_torch, melgan, os.path.join(hp.synth_path, 'step_{}_{}.wav'.format(current_step, hp.vocoder)))
                        utils.melgan_infer(mel_postnet_torch, melgan, os.path.join(hp.synth_path, 'step_{}_postnet_{}.wav'.format(current_step, hp.vocoder)))
                        utils.melgan_infer(mel_target_torch, melgan, os.path.join(hp.synth_path, 'step_{}_ground-truth_{}.wav'.format(current_step, hp.vocoder)))
                    elif hp.vocoder == 'waveglow':
                        utils.waveglow_infer(mel_torch, waveglow, os.path.join(hp.synth_path, 'step_{}_{}.wav'.format(current_step, hp.vocoder)))
                        utils.waveglow_infer(mel_postnet_torch, waveglow, os.path.join(hp.synth_path, 'step_{}_postnet_{}.wav'.format(current_step, hp.vocoder)))
                        utils.waveglow_infer(mel_target_torch, waveglow, os.path.join(hp.synth_path, 'step_{}_ground-truth_{}.wav'.format(current_step, hp.vocoder)))

                
                if current_step % hp.eval_step == 0:
                    model.eval()
                    with torch.no_grad():
                        d_l, m_l, m_p_l = evaluate(model, current_step)
                        t_l = d_l + m_l + m_p_l
                        
                        val_logger.add_scalar('Loss/total_loss', t_l, current_step)
                        val_logger.add_scalar('Loss/mel_loss', m_l, current_step)
                        val_logger.add_scalar('Loss/mel_postnet_loss', m_p_l, current_step)
                        val_logger.add_scalar('Loss/duration_loss', d_l, current_step)

                    model.train()

                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    args = parser.parse_args()

    main(args)
