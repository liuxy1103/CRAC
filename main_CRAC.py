from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import yaml
import time
import cv2
import h5py
import random
import logging
import argparse
import numpy as np
from PIL import Image
from attrdict import AttrDict
from tensorboardX import SummaryWriter
from collections import OrderedDict
import multiprocessing as mp
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model.SCPN import SCPNNetwork
from data_provider_labeled import Provider
from provider_valid import Provider_valid
from loss.loss import WeightedMSE, WeightedBCE
from loss.loss import MSELoss, BCELoss
from utils.show import show_affs, show_affs_whole
from unet3d_mala import UNet3D_MALA_embedding as UNet3D_MALA
from model_superhuman2 import UNet_PNI_embedding as UNet_PNI
from unet3d_mala import UNet3D_MALA as UNet3D_MALA_aff
from model_superhuman2 import UNet_PNI as UNet_PNI_aff
from utils.utils import setup_seed, execute
from utils.shift_channels import shift_func
from loss.loss_embedding_mse import embedding_loss_norm1, embedding_loss_norm5,embedding_loss_norm_multi_relu,embedding_loss_norm_multi

import waterz
from utils.lmc import mc_baseline
from utils.fragment import watershed, randomlabel
# import evaluate as ev
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import warnings
warnings.filterwarnings("ignore")

def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
                level    = logging.INFO,
                format   = '%(message)s',
                datefmt  = '%m-%d %H:%M',
                filename = path,
                filemode = 'w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    # seeds
    setup_seed(cfg.TRAIN.random_seed)
    if cfg.TRAIN.if_cuda:
        if torch.cuda.is_available() is False:
            raise AttributeError('No GPU available')

    prefix = cfg.time
    if cfg.TRAIN.resume:
        model_name = cfg.TRAIN.model_name
    else:
        model_name = prefix + '_' + cfg.NAME
    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, model_name)
    cfg.cache_path2 = os.path.join(cfg.TRAIN.cache_path2, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    # cfg.record_path = os.path.join(cfg.TRAIN.record_path, 'log')
    cfg.record_path = os.path.join(cfg.save_path, model_name)
    cfg.valid_path = os.path.join(cfg.save_path, 'valid')
    if cfg.TRAIN.resume is False:
        if not os.path.exists(cfg.cache_path):
            os.makedirs(cfg.cache_path)
        if not os.path.exists(cfg.cache_path2):
            os.makedirs(cfg.cache_path2)
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)
        if not os.path.exists(cfg.record_path):
            os.makedirs(cfg.record_path)
        if not os.path.exists(cfg.valid_path):
            os.makedirs(cfg.valid_path)
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    return writer

def load_dataset(cfg):
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    if cfg.TRAIN.if_valid:
        valid_provider = Provider_valid(cfg)
    else:
        valid_provider = None
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider

def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    if cfg.MODEL.model_type == 'mala':
        print('load mala model!')
        model = UNet3D_MALA(output_nc=cfg.MODEL.output_nc,
                            if_sigmoid=cfg.MODEL.if_sigmoid,
                            init_mode=cfg.MODEL.init_mode_mala,
                            emd=cfg.MODEL.emd).to(device)
    else:
        print('load superhuman model!')
        model = UNet_PNI(in_planes=cfg.MODEL.input_nc,
                        out_planes=cfg.MODEL.output_nc,
                        filters=cfg.MODEL.filters,
                        upsample_mode=cfg.MODEL.upsample_mode,
                        decode_ratio=cfg.MODEL.decode_ratio,
                        merge_mode=cfg.MODEL.merge_mode,
                        pad_mode=cfg.MODEL.pad_mode,
                        bn_mode=cfg.MODEL.bn_mode,
                        relu_mode=cfg.MODEL.relu_mode,
                        init_mode=cfg.MODEL.init_mode,
                        emd=cfg.MODEL.emd).to(device)

    if cfg.MODEL.pre_train:
        print('Load pre-trained model ...')
        ckpt_path = os.path.join('../models', \
            cfg.MODEL.trained_model_name, \
            'model-%06d.ckpt' % cfg.MODEL.trained_model_id)
        checkpoint = torch.load(ckpt_path)
        pretrained_dict = checkpoint['model_weights']
        if cfg.MODEL.trained_gpus > 1:
            pretained_model_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                name = k[7:] # remove module.
                # name = k
                pretained_model_dict[name] = v
        else:
            pretained_model_dict = pretrained_dict

        from utils.encoder_dict import ENCODER_DICT2, ENCODER_DECODER_DICT2
        model_dict = model.state_dict()
        encoder_dict = OrderedDict()
        if cfg.MODEL.if_skip == 'True':
            print('Load the parameters of encoder and decoder!')
            encoder_dict = {k: v for k, v in pretained_model_dict.items() if k.split('.')[0] in ENCODER_DECODER_DICT2}
        else:
            print('Load the parameters of encoder!')
            encoder_dict = {k: v for k, v in pretained_model_dict.items() if k.split('.')[0] in ENCODER_DICT2}
        model_dict.update(encoder_dict)
        model.load_state_dict(model_dict)

    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model

def build_model_aff(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    if cfg.MODEL.model_type == 'mala':
        print('load mala model!')
        model = UNet3D_MALA_aff(output_nc=cfg.MODEL.output_nc, if_sigmoid=cfg.MODEL.if_sigmoid, init_mode=cfg.MODEL.init_mode_mala).to(device)
    else:
        print('load superhuman model!')
        model = UNet_PNI_aff(in_planes=cfg.MODEL.input_nc,
                        out_planes=cfg.MODEL.output_nc,
                        filters=cfg.MODEL.filters,
                        upsample_mode=cfg.MODEL.upsample_mode,
                        decode_ratio=cfg.MODEL.decode_ratio,
                        merge_mode=cfg.MODEL.merge_mode,
                        pad_mode=cfg.MODEL.pad_mode,
                        bn_mode=cfg.MODEL.bn_mode,
                        relu_mode=cfg.MODEL.relu_mode,
                        init_mode=cfg.MODEL.init_mode).to(device)

    if cfg.MODEL.pre_train:
        print('Load pre-trained model ...')
        ckpt_path = os.path.join('../models', \
            cfg.MODEL.trained_model_name, \
            'model-%06d.ckpt' % cfg.MODEL.trained_model_id)
        checkpoint = torch.load(ckpt_path)
        pretrained_dict = checkpoint['model_weights']
        if cfg.MODEL.trained_gpus > 1:
            pretained_model_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                name = k[7:] # remove module.
                # name = k
                pretained_model_dict[name] = v
        else:
            pretained_model_dict = pretrained_dict

        from utils.encoder_dict import ENCODER_DICT2, ENCODER_DECODER_DICT2
        model_dict = model.state_dict()
        encoder_dict = OrderedDict()
        if cfg.MODEL.if_skip == 'True':
            print('Load the parameters of encoder and decoder!')
            encoder_dict = {k: v for k, v in pretained_model_dict.items() if k.split('.')[0] in ENCODER_DECODER_DICT2}
        else:
            print('Load the parameters of encoder!')
            encoder_dict = {k: v for k, v in pretained_model_dict.items() if k.split('.')[0] in ENCODER_DICT2}
        model_dict.update(encoder_dict)
        model.load_state_dict(model_dict)

    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model

def build_model_SCPN(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    global  SCPNN
    SCPNN = SCPNNetwork().cuda()

    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            SCPNN = nn.DataParallel( SCPNN)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)



    print('Load pretrained SCPN Model')
    model_path = os.path.join(cfg.TRAIN.model_SCPN_path, 'model-SCPN-%06d.ckpt' % cfg.TRAIN.SCPN_model_id)

    if os.path.isfile(model_path): 
        checkpoint = torch.load(model_path)
        SCPNN.load_state_dict(checkpoint['model_weights'])
        # optimizer.load_state_dict(checkpoint['optimizer_weights'])
        
    else:
        raise AttributeError('No checkpoint found at %s' % model_path)
    print('Done (time: %.2fs)' % (time.time() - t1))
    print('valid %d' % checkpoint['current_iter'])
    for k, v in SCPNN.named_parameters():
        v.requires_grad = False
    return SCPNN

def resume_params(cfg, model, optimizer, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.save_path, 'model-%06d.ckpt' % cfg.TRAIN.model_id)

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_weights'])
            # optimizer.load_state_dict(checkpoint['optimizer_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        return model, optimizer, checkpoint['current_iter']
    else:
        return model, optimizer, 0

def resume_params_aff(cfg, model, optimizer, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.save_path, 'model_aff-%06d.ckpt' % cfg.TRAIN.model_id)

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_weights'])
            # optimizer.load_state_dict(checkpoint['optimizer_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        return model, optimizer, checkpoint['current_iter']
    else:
        return model, optimizer, 0


def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr


def loop(cfg, train_provider, valid_provider, model,model2, criterion, optimizer, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0
    sum_loss_embedding = 0
    sum_loss_embedding_pseudo = 0
    sum_loss_emb_consistency = 0
    device = torch.device('cuda:0')
    
    if cfg.TRAIN.loss_func == 'MSELoss':
        criterion = MSELoss()
    elif cfg.TRAIN.loss_func == 'BCELoss':
        criterion = BCELoss()
    elif cfg.TRAIN.loss_func == 'WeightedBCELoss':
        criterion = WeightedBCE()
    elif cfg.TRAIN.loss_func == 'WeightedMSELoss':
        criterion = WeightedMSE()
    else:
        raise AttributeError("NO this criterion")
    criterion_ct = MSELoss()
    while iters <= cfg.TRAIN.total_iters:
        # train
        model.train()
        model2.train()
        # SCPN.eval()
        iters += 1
        t1 = time.time()
        inputs, lb, target, weightmap = train_provider.next()
        
        # decay learning rate
        if cfg.TRAIN.end_lr == cfg.TRAIN.base_lr:
            current_lr = cfg.TRAIN.base_lr
        else:
            current_lr = calculate_lr(iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        optimizer.zero_grad()
        embedding = model(inputs)
        pred2 = model2(inputs)

        ##############################
        # LOSS
        # loss = criterion(pred, target, weightmap)
        if cfg.DATA.if_sparse:
            mask_instance_labeled = lb>0
            mask_border_labeled = target==0
            mask_label = mask_instance_labeled.unsqueeze(1) + mask_border_labeled
            mask_label = mask_label>0
            mask_label = mask_label.float()

        loss_emb1, pred = embedding_loss_norm_multi(embedding, target, weightmap*mask_label, criterion, affs0_weight=cfg.TRAIN.affs0_weight,shift=cfg.DATA.shift_channels)
        weightmap_tmp = weightmap
        weightmap_tmp[:,:3] = weightmap_tmp[:,:3]*cfg.TRAIN.affs0_weight
        loss_emb2 = criterion(pred2, target, weightmap_tmp*mask_label)        

        loss_emb = (loss_emb1+loss_emb2) * cfg.TRAIN.emb_weight
        # print('loss_emb:', loss_emb)

        #entropy map2
        pred_tmp = F.relu(pred)
        entropy = -(pred_tmp * torch.log(pred_tmp + 1e-10)+ (1-pred_tmp)*torch.log(1-pred_tmp+ 1e-10))
        entropy2 = -(pred2 * torch.log(pred2 + 1e-10)+ (1-pred2)*torch.log(1-pred2+ 1e-10))
        # alpha_t = cfg.TRAIN.aff_thres
        
        # thres_mask = (pred>cfg.TRAIN.aff_thres).float() + (pred<1-cfg.TRAIN.aff_thres).float()
        # thres_mask2 = (pred2>cfg.TRAIN.aff_thres).float() + (pred2<1-cfg.TRAIN.aff_thres).float()
        # print(pred_tmp.min(),pred_tmp.max(),torch.mean(pred_tmp))
        # print(pred2.min(),pred2.max(),torch.mean(pred2))
        # print(entropy.min(),entropy.max(),torch.mean(entropy))
        # print(entropy2.min(),entropy2.max(),torch.mean(entropy2))
        alpha_t = cfg.TRAIN.alpha0 * pow(1-float(iters) / cfg.TRAIN.total_iters, cfg.TRAIN.power)  # 越来越小
        # print(alpha_t)
        gamma_t=np.percentile(entropy.flatten().cpu().detach().numpy(),100*(1-alpha_t)) #gamma_t 越来越大
        gamma_t2=np.percentile(entropy2.flatten().cpu().detach().numpy(),100*(1-alpha_t))
        # print('gamma_t',gamma_t)
        # print('gamma_t2',gamma_t2)
        thres_mask2_1 = (entropy<gamma_t).float() 
        thres_mask2_2 = (entropy2<gamma_t2).float()

        gate = 5000
        if iters == gate:
            _ = build_model_SCPN(cfg, writer)
        if iters > gate:
            SCPN = SCPNN
            SCPN.eval()
            #entropy map
            pred_tmp = F.relu(pred)
            entropy = -(pred_tmp * torch.log(pred_tmp + 1e-10)+ (1-pred_tmp)*torch.log(1-pred_tmp+ 1e-10))
            entropy2 = -(pred2 * torch.log(pred2 + 1e-10)+ (1-pred2)*torch.log(1-pred2+ 1e-10))

            #entropy map1
            pred = F.relu(pred)
            error_map = torch.zeros_like(pred)
            entropy = -(pred * torch.log(pred + 1e-10)+ (1-pred)*torch.log(1-pred+ 1e-10))
            for i in range(error_map.shape[0]):
                error_map_tmp = SCPN(inputs, pred[:,i], entropy[:,i])
                error_map[:,i:i+1] = error_map_tmp

            pred_bi = torch.zeros_like(pred_tmp)
            pred_bi[pred_tmp>= 0.5] = 1
            pred_bi[pred_tmp< 0.5] = 0
            error_map_gt = abs(pred_bi-target)  # 0 is right 1 is false
            error_map[error_map>= 0.5] = 1
            error_map[error_map< 0.5] = 0
            thres_mask1_1 = 1- error_map

            error_map2 = torch.zeros_like(pred_tmp)
            for i in range(error_map.shape[0]):
                error_map_tmp2 = SCPN(inputs, pred2[:,i], entropy2[:,i])
                error_map2[:,i:i+1] = error_map_tmp2
            error_map2[error_map2>= 0.5] = 1
            error_map2[error_map2< 0.5] = 0
            thres_mask1_2 = 1- error_map2
            thres_mask = thres_mask1_1 * thres_mask2_1
            thres_mask2 = thres_mask1_2 * thres_mask2_2 
        else:
            thres_mask =  thres_mask2_1
            thres_mask2 =  thres_mask2_2


        loss_emb_psudo1, _ = embedding_loss_norm_multi(embedding, pred2, weightmap*(1-mask_label)*thres_mask2, criterion, affs0_weight=cfg.TRAIN.affs0_weight,shift=cfg.DATA.shift_channels)
        loss_emb_psudo2 = criterion(pred2, pred, weightmap_tmp*(1-mask_label)*thres_mask)
        loss_emb_pseudo = (loss_emb_psudo1+loss_emb_psudo2)*cfg.TRAIN.pseudo_weight
        loss_emb_consistency =  0
        loss = loss_emb + loss_emb_pseudo 
        loss.backward()
        shift = 1
        pred[:, 1, :, :shift, :] = pred[:, 1, :, shift:shift*2, :]
        pred[:, 2, :, :, :shift] = pred[:, 2, :, :, shift:shift*2]
        pred[:, 0, :shift, :, :] = pred[:, 0, shift:shift*2, :, :]
        pred = F.relu(pred[:, :3])
        ##############################

        if cfg.TRAIN.weight_decay is not None:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)
        optimizer.step()
        
        sum_loss += loss.item()
        sum_loss_embedding_pseudo += loss_emb_pseudo.item()
        sum_loss_embedding += loss_emb.item()
        sum_loss_emb_consistency += loss_emb_consistency
        sum_time += time.time() - t1
        
        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info('step %d, loss=%.6f, loss_embedding=%.6f, loss_embedding_pseudo=%.6f, loss_emb_consistency=%.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss * 1,sum_loss_embedding,sum_loss_embedding_pseudo,sum_loss_emb_consistency, current_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss * 1, iters)
            else:
                logging.info('step %d, loss=%.6f, loss_embedding=%.6f, loss_embedding_pseudo=%.6f, loss_emb_consistency=%.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss / cfg.TRAIN.display_freq * 1, sum_loss_embedding / cfg.TRAIN.display_freq * 1, sum_loss_embedding_pseudo/ cfg.TRAIN.display_freq * 1,
                            sum_loss_emb_consistency / cfg.TRAIN.display_freq * 1, current_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))

                writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq * 1, iters)
                writer.add_scalar('loss_embedding', sum_loss_embedding / cfg.TRAIN.display_freq * 1, iters)
                writer.add_scalar('loss_embedding_pseudo', sum_loss_embedding_pseudo / cfg.TRAIN.display_freq * 1, iters)
                writer.add_scalar('loss_emb_consistency', sum_loss_emb_consistency / cfg.TRAIN.display_freq * 1, iters)
            f_loss_txt.write('step = %d, loss = %.6f, loss_embedding = %.6f, loss_embedding_pseudo = %.6f, loss_embedding_consistency = %.6f' % \
                (iters, sum_loss / cfg.TRAIN.display_freq, sum_loss_embedding / cfg.TRAIN.display_freq,\
                 sum_loss_embedding_pseudo / cfg.TRAIN.display_freq, sum_loss_emb_consistency / cfg.TRAIN.display_freq))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0 
            sum_loss = 0
            sum_loss_embedding_pseudo = 0
            sum_loss_emb_consistency = 0
            sum_loss_embedding = 0

        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            if iters > gate:
                show_affs(iters, inputs, error_map[:,:3], error_map_gt[:,:3], cfg.cache_path2, model_type=cfg.MODEL.model_type)

            show_affs(iters, inputs, pred[:,:3], target[:,:3], cfg.cache_path, model_type=cfg.MODEL.model_type)

        # valid
        if cfg.TRAIN.if_valid:
            if iters % cfg.TRAIN.save_freq == 0 or iters == 1:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model.eval()
                model2.eval()
                dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                                shuffle=False, drop_last=False, pin_memory=True)
                losses_valid = []
                for k, batch in enumerate(dataloader, 0):
                    inputs, target, weightmap = batch
                    inputs = inputs.cuda()
                    target = target.cuda()
                    weightmap = weightmap.cuda()
                    if cfg.DATA.if_sparse:
                        mask_instance_labeled = lb>0
                        mask_border_labeled = target==0
                        mask_label = mask_instance_labeled.unsqueeze(1) + mask_border_labeled
                        mask_label = mask_label>0
                        mask_label = mask_label.float()
                    with torch.no_grad():
                        embedding = model(inputs)
                        pred2 = model2(inputs)
                    loss_emb1, pred = embedding_loss_norm_multi(embedding, target, weightmap*mask_label, criterion, affs0_weight=cfg.TRAIN.affs0_weight,shift=cfg.DATA.shift_channels)
                    weightmap_tmp = weightmap
                    weightmap_tmp[:,:3] = weightmap_tmp[:,:3]*cfg.TRAIN.affs0_weight
                    loss_emb2 = criterion(pred2, target, weightmap_tmp*mask_label)        


                    thres_mask = (pred>cfg.TRAIN.aff_thres).float() + (pred<1-cfg.TRAIN.aff_thres).float()

                    thres_mask2 = (pred2>cfg.TRAIN.aff_thres).float() + (pred2<1-cfg.TRAIN.aff_thres).float()
                    

                    loss_emb = (loss_emb1+loss_emb2) * cfg.TRAIN.emb_weight
                    loss_emb_psudo1, _ = embedding_loss_norm_multi(embedding, pred2, weightmap*(1-mask_label)*thres_mask2, criterion, affs0_weight=cfg.TRAIN.affs0_weight,shift=cfg.DATA.shift_channels)
                    loss_emb_psudo2= criterion(pred2, pred, weightmap_tmp*(1-mask_label)*thres_mask)
                    loss_emb_pseudo = (loss_emb_psudo1+loss_emb_psudo2)*cfg.TRAIN.pseudo_weight
                    loss_emb_consistency =  0
                    tmp_loss = loss_emb + loss_emb_pseudo

                    shift = 1
                    pred[:, 1, :, :shift, :] = pred[:, 1, :, shift:shift*2, :]
                    pred[:, 2, :, :, :shift] = pred[:, 2, :, :, shift:shift*2]
                    pred[:, 0, :shift, :, :] = pred[:, 0, shift:shift*2, :, :]
                    pred = F.relu(pred)
                    losses_valid.append(tmp_loss.item())
                    valid_provider.add_vol(np.squeeze(pred.data.cpu().numpy()))
                epoch_loss = sum(losses_valid) / len(losses_valid)
                out_affs = valid_provider.get_results()
                gt_affs = valid_provider.get_gt_affs().copy()
                gt_seg = valid_provider.get_gt_lb()
                valid_provider.reset_output()
                out_affs = out_affs[:3]
                # gt_affs = gt_affs[:, :3]
                show_affs_whole(iters, out_affs, gt_affs, cfg.valid_path)

                ##############
                # segmentation
                if cfg.TRAIN.if_seg:
                    if iters > 5000:
                        fragments = watershed(out_affs, 'maxima_distance')
                        sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
                        seg_waterz = list(waterz.agglomerate(out_affs, [0.50],
                                    fragments=fragments,
                                    scoring_function=sf,
                                    discretize_queue=256))[0]
           
                        # sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
   
                        arand_waterz = adapted_rand_ref(gt_seg, seg_waterz, ignore_labels=(0))[0]
                        voi_split_waterz, voi_merge_waterz = voi_ref(gt_seg, seg_waterz, ignore_labels=(0))
                        voi_sum_waterz = voi_split_waterz + voi_merge_waterz

                        seg_lmc = mc_baseline(out_affs)
                        arand_lmc = adapted_rand_ref(gt_seg, seg_lmc, ignore_labels=(0))[0]
                        voi_split_lmc, voi_merge_lmc = voi_ref(gt_seg, seg_lmc, ignore_labels=(0))
                        voi_sum_lmc = voi_split_lmc + voi_merge_lmc
                    else:
                        voi_split_waterz = 0.0
                        voi_merge_waterz = 0.0
                        voi_split_lmc = 0.0
                        voi_merge_lmc = 0.0
                        voi_sum_waterz = 0.0
                        arand_waterz = 0.0
                        voi_sum_lmc = 0.0
                        arand_lmc = 0.0
                        print('model-%d, segmentation failed!' % iters)
                else:
                    voi_split_waterz = 0.0
                    voi_merge_waterz = 0.0
                    voi_split_lmc = 0.0
                    voi_merge_lmc = 0.0
                    voi_sum_waterz = 0.0
                    arand_waterz = 0.0
                    voi_sum_lmc = 0.0
                    arand_lmc = 0.0
                ##############

                # MSE
                whole_mse = np.sum(np.square(out_affs - gt_affs)) / np.size(gt_affs)
                out_affs = np.clip(out_affs, 0.000001, 0.999999)
                bce = -(gt_affs * np.log(out_affs) + (1 - gt_affs) * np.log(1 - out_affs))
                whole_bce = np.sum(bce) / np.size(gt_affs)
                out_affs[out_affs <= 0.5] = 0
                out_affs[out_affs > 0.5] = 1
                # whole_f1 = 1 - f1_score(gt_affs.astype(np.uint8).flatten(), out_affs.astype(np.uint8).flatten())
                whole_f1 = f1_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - out_affs.astype(np.uint8).flatten())
                print('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, voi_split_waterz=%.6f,\
                 voi_merge_waterz=%.6f, VOI-waterz=%.6f, ARAND-waterz=%.6f, voi_split_lmc=%.6f, voi_merge_lmc=%.6f, VOI-lmc=%.6f, ARAND-lmc=%.6f' % \
                    (iters, epoch_loss, whole_mse, whole_bce, whole_f1,voi_split_waterz,voi_merge_waterz, voi_sum_waterz,\
                         arand_waterz, voi_split_lmc,voi_merge_lmc, voi_sum_lmc, arand_lmc), flush=True)
                writer.add_scalar('valid/epoch_loss', epoch_loss, iters)
                writer.add_scalar('valid/mse_loss', whole_mse, iters)
                writer.add_scalar('valid/bce_loss', whole_bce, iters)
                writer.add_scalar('valid/f1_score', whole_f1, iters)
                writer.add_scalar('valid/voi_waterz', voi_sum_waterz, iters)
                writer.add_scalar('valid/arand_waterz', arand_waterz, iters)
                writer.add_scalar('valid/voi_lmc', voi_sum_lmc, iters)
                writer.add_scalar('valid/arand_lmc', arand_lmc, iters)
                f_valid_txt.write('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, voi_split_waterz=%.6f,\
                 voi_merge_waterz=%.6f, VOI-waterz=%.6f, ARAND-waterz=%.6f, voi_split_lmc=%.6f, voi_merge_lmc=%.6f, VOI-lmc=%.6f, ARAND-lmc=%.6f' % \
                    (iters, epoch_loss, whole_mse, whole_bce, whole_f1,voi_split_waterz,voi_merge_waterz, voi_sum_waterz,\
                         arand_waterz, voi_split_lmc,voi_merge_lmc, voi_sum_lmc, arand_lmc))
                f_valid_txt.write('\n')
                f_valid_txt.flush()
                torch.cuda.empty_cache()

        # save
        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                    'model_weights': model.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'model-%06d.ckpt' % iters))
            print('***************save modol, iters = %d.***************' % (iters), flush=True)
        # save
        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                    'model_weights': model2.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'model_aff-%06d.ckpt' % iters))
            print('***************save modol, iters = %d.***************' % (iters), flush=True)
    f_loss_txt.close()
    f_valid_txt.close()


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='seg_inpainting', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))

    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    cfg.path = cfg_file
    cfg.time = time_stamp
    if cfg.DATA.shift_channels is None:
        # assert cfg.MODEL.output_nc == 3, "output_nc must be 3"
        cfg.shift = None
    else:
        assert cfg.MODEL.output_nc == len(cfg.DATA.shift_channels), "output_nc must be equal to shift_channels"
        # cfg.shift = shift_func(cfg.DATA.shift_channels)

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider, valid_provider = load_dataset(cfg)
        model = build_model(cfg, writer)
        model2 = build_model_aff(cfg, writer)
        

        optimizer = torch.optim.Adam([{'params': model.parameters()},
                                    {'params': model2.parameters()}], lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
                                 eps=0.01, weight_decay=1e-6, amsgrad=True)
        # optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
        # optimizer = optim.Adamax(model.parameters(), lr=cfg.TRAIN.base_l, eps=1e-8)
        model, optimizer, init_iters = resume_params(cfg, model, optimizer, cfg.TRAIN.resume)
        model2, optimizer, init_iters = resume_params_aff(cfg, model2, optimizer, cfg.TRAIN.resume)
        _ = build_model_SCPN(cfg, writer)
        loop(cfg, train_provider, valid_provider, model, model2, nn.L1Loss(), optimizer, init_iters, writer)

        writer.close()
    else:
        pass
    print('***Done***')