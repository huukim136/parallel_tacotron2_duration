import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env # checked
from meldataset import MelDataset, get_dataset_filelist, mel_spectrogram # checked
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss # checked
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint, scan_checkpoint_tts # checked
from utils_pas import plot_alignment_to_numpy
torch.backends.cudnn.benchmark = True

########### TTS #############
from PASpeech2.fastspeech2 import FastSpeech2
from PASpeech2.loss import FastSpeech2Loss
# from PASpeech2 import hparams as hp
import hparams as hp
import utils_pas
import numpy as np
import random
import pdb
############################

def plot_attn(train_logger, C, current_step, hp):
    # pdb.set_trace()
    idx = random.randint(0, C.size(0) - 1)   

    train_logger.add_image(
        "encoder.C",
        plot_alignment_to_numpy(C.data.cpu().numpy()[idx].T),
        current_step)

    # train_logger.add_image(
    #     "encoder.E",
    #     plot_alignment_to_numpy(E.data.cpu().numpy()[idx].T),
    #     current_step)

    # train_logger.add_image(
    #     "encoder.W",
    #     plot_alignment_to_numpy(W.data.cpu().numpy()[idx].T),
    #     current_step)

def train(rank, a, h):
    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    model_tts = FastSpeech2().to(device) # TTS on device

    if rank == 0:
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
        print('checkpoint not found. begin from 0')
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        model_tts.load_state_dict(state_dict_g['model'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
        print('checkpoint loaded from ', a.checkpoint_path)

    optim_g = torch.optim.AdamW(itertools.chain(generator.parameters(), model_tts.parameters())
                                , h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters())
                                , h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    
    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    ##
    torch.nn.utils.clip_grad_norm_(generator.parameters(),1.0)
    torch.nn.utils.clip_grad_norm_(mpd.parameters(),1.0)
    torch.nn.utils.clip_grad_norm_(msd.parameters(),1.0)
    torch.nn.utils.clip_grad_norm_(model_tts.parameters(), 1.0) # TTS    
    ##

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    tts_loss = FastSpeech2Loss().to(device)

    training_data, validation_data = get_dataset_filelist(a)
    trainset = MelDataset(training_data, h.segment_size, h.num_mels, h.sampling_rate, n_cache_reuse=0, shuffle= True, device=device)
    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=True, sampler=None, 
                              collate_fn=training_data.collate_fn, batch_size=hp.batch_size, pin_memory=True, drop_last=True)

    if rank == 0:
        validset = MelDataset(validation_data, h.segment_size, h.num_mels, h.sampling_rate, False, False, n_cache_reuse=0, device=device)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False, sampler=None, 
                                       collate_fn=validation_data.collate_fn, batch_size=1, pin_memory=True, drop_last=True)
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    # training
    generator.train()
    mpd.train()
    msd.train()
    model_tts.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        for i, batch in enumerate(train_loader):
            if rank == 0: start_b = time.time()
            y, tts_data, audio_start = batch
            ### for TTS data
            text = torch.from_numpy(tts_data['text']).long().to(device)

            D = torch.from_numpy(tts_data['D']).long().to(device)
            log_D = torch.from_numpy(tts_data['log_D']).float().to(device)
            f0 = torch.from_numpy(tts_data['f0']).float().to(device)
            energy = torch.from_numpy(tts_data['energy']).float().to(device)

            src_len = torch.from_numpy(tts_data['src_len']).long().to(device)
            mel_len = torch.from_numpy(tts_data['mel_len']).long().to(device)
            max_src_len = np.max(tts_data['src_len']).astype(np.int32)
            max_mel_len = np.max(tts_data['mel_len']).astype(np.int32)

            ###### TTS forward
            y = torch.stack(y) # batch, segment length
            y = torch.autograd.Variable(y.to(device, non_blocking=True)) # wav to device
            y = y.unsqueeze(1)

            # text -> [encoder] -> [var_adapt] -> var_output + pos_emb => mel_output
            # mel_output, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _, S, E = model_tts(
            #     text, src_len, mel_len, D, f0, energy, max_src_len, max_mel_len)
            mel_output, d, f0_output, energy_output, src_mask, mel_mask, _, C = model_tts(
                text, src_len, mel_len, D, f0, energy, max_src_len, max_mel_len)
            plot_attn(sw, C, steps, hp)

            # sync
            indices = torch.tensor(audio_start)
            x_hat = []
            for i, idx in enumerate(indices):
                chunk = mel_output[i,idx:idx + y.size(-1)//hp.hop_length,:] # actually, var + text_emb
                x_hat.append(chunk)
            x_hat = torch.stack(x_hat).transpose(1,2)
            y_g_hat = generator(x_hat) # use output of TTS

            y_mel = mel_spectrogram(y.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)

            ## Discriminator
            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()
           

            ## Generator
            optim_g.zero_grad()

            ### include TTS parts
            # Cal loss
            d_loss, f_loss, e_loss = tts_loss(d, f0_output, f0, energy_output, energy, ~src_mask, ~mel_mask)
            total_tts_loss = d_loss + f_loss + e_loss # actually variance adaptor error

            t_l = total_tts_loss.item()
            d_l = d_loss.item()
            f_l = f_loss.item()
            e_l = e_loss.item()
            
            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel \
                           + total_tts_loss # added for TTS

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print('GAN Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, mel_error, time.time() - start_b))
                    print('TTS steps : Total loss: {:.4f}, Duration loss: {:.4f}, F0 loss: {:.4f}, E loss: {:.4f}'.
                          format(t_l, d_l, f_l, e_l))
                    print()

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': generator.state_dict(),
                                     'model': model_tts.state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {'mpd': mpd.state_dict(),
                                     'msd': msd.state_dict(),
                                     'optim_g': optim_g.state_dict(), 
                                     'optim_d': optim_d.state_dict(),
                                     'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    sw.add_scalar("training/pitch_error", f_loss, steps)
                    sw.add_scalar("training/energy_error", e_loss, steps)
                    sw.add_scalar("training/duration_error", d_loss, steps)

                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    model_tts.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            y, val_tts_data, _ = batch
                            ##########
                            v_id = val_tts_data['id']
                            v_text = torch.from_numpy(val_tts_data['text']).long().to(device)
                            v_D = torch.from_numpy(val_tts_data['D']).long().to(device)
                            v_log_D = torch.from_numpy(val_tts_data['log_D']).float().to(device)
                            v_f0 = torch.from_numpy(val_tts_data['f0']).float().to(device)
                            v_energy = torch.from_numpy(val_tts_data['energy']).float().to(device)
                            v_src_len = torch.from_numpy(val_tts_data['src_len']).long().to(device)
                            v_mel_len = torch.from_numpy(val_tts_data['mel_len']).long().to(device)
                            v_max_src_len = np.max(val_tts_data['src_len']).astype(np.int32)
                            v_max_mel_len = np.max(val_tts_data['mel_len']).astype(np.int32)
                            y = torch.stack(y)
                            v_mel_output, v_d, v_f0_output, v_energy_output, v_src_mask, v_mel_mask, v_out_mel_len, C= model_tts(
                                v_text, v_src_len, v_mel_len, v_D, v_f0, v_energy, v_max_src_len, v_max_mel_len)
                            ##########
                            y_g_hat = generator(v_mel_output.transpose(1,2))
                            y_mel = mel_spectrogram(y.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
                            ##########
                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_image('gt/y_spec_{}'.format(j), plot_alignment_to_numpy(y_mel[0].cpu()), steps)
                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
                                sw.add_image('generated/y_hat_spec_{}'.format(j), plot_alignment_to_numpy(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                    generator.train()
                    model_tts.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

def main():
    print('Initializing Training Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs_float') # convert to float32
    parser.add_argument('--input_training_file', default='train.txt')
    parser.add_argument('--input_validation_file', default='val.txt')
    parser.add_argument('--checkpoint_path', default='/home/hk/log_and_save/hispeech/cp_duration_EATS')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=15000, type=int)
    parser.add_argument('--summary_interval', default=200, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    a = parser.parse_args()

    with open(a.config) as f: data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else: pass
    train(0, a, h)

if __name__ == '__main__':
    main()
