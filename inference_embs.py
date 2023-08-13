import os
import cv2
import h5py
import yaml
import torch
import argparse
import numpy as np
from skimage import morphology
from attrdict import AttrDict
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from provider_valid_inference import Provider_valid
from loss.loss import BCELoss, WeightedBCE, MSELoss, WeightedMSE
from unet3d_mala import UNet3D_MALA_embedding as UNet3D_MALA
from utils.shift_channels import shift_func
from loss.loss_embedding_mse import embedding_loss_norm1, embedding_loss_norm5,embedding_loss_norm_multi_relu,embedding_loss_norm_multi
from model_superhuman2 import UNet_PNI_embedding as UNet_PNI
import waterz
from utils.lmc import mc_baseline
from utils.fragment import watershed, randomlabel, relabel
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from skimage import io

def embedding_pca(embeddings, n_components=3, as_rgb=False):
    if as_rgb and n_components != 3:
        raise ValueError("")

    pca = PCA(n_components=n_components)
    embed_dim = embeddings.shape[0]
    shape = embeddings.shape[1:]

    embed_flat = embeddings.reshape(embed_dim, -1).T
    embed_flat = pca.fit_transform(embed_flat).T
    embed_flat = embed_flat.reshape((n_components,) + shape)

    if as_rgb:
        embed_flat = 255 * (embed_flat - embed_flat.min()) / np.ptp(embed_flat)
        embed_flat = embed_flat.astype('uint8')
    return embed_flat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='data75_Cremi_sparse5_percent0.005_SCPN0.1_inf_64k_ensemble_Test-Cremi_A', help='path to config file')
    parser.add_argument('-mn', '--model_name', type=str, default='2022-10-11--16-17-06_data75_Cremi_sparse10_long234927_aff10_embw10_psdw1_ctw0_batch2_crossori_thresAuto_modeling_entropy_percent0.005_v3')
    parser.add_argument('-emb', '--if_embedding', action='store_true', default=True)
    parser.add_argument('-id', '--model_id', type=int, default=122000)
    parser.add_argument('-m', '--mode', type=str, default='Cremi-A')
    parser.add_argument('-ts', '--test_split', type=int, default=20)
    parser.add_argument('-pm', '--pixel_metric', action='store_true', default=True)
    parser.add_argument('-sw', '--show', action='store_true', default=True)
    parser.add_argument('-lt', '--lmc_thres', type=float, default=0.36)
    parser.add_argument('-sa', '--save_affs', action='store_true', default=False)#
    parser.add_argument('-mutex', '--mutex', action='store_true', default=False)
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))

    if args.model_name is not None:
        trained_model = args.model_name
    else:
        trained_model = cfg.TEST.model_name

    out_path = os.path.join('../inference', trained_model, args.mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_folder = 'affs_'+str(args.model_id)
    out_affs = os.path.join(out_path, img_folder)
    if not os.path.exists(out_affs):
        os.makedirs(out_affs)
    print('out_path: ' + out_affs)
    affs_img_path = os.path.join(out_affs, 'affs_img')
    seg_img_path = os.path.join(out_affs, 'seg_img')
    if not os.path.exists(affs_img_path):
        os.makedirs(affs_img_path)
    if not os.path.exists(seg_img_path):
        os.makedirs(seg_img_path)

    device = torch.device('cuda:0')


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

    ckpt_path = os.path.join('../models', trained_model, 'model-%06d.ckpt' % args.model_id)
    checkpoint = torch.load(ckpt_path)

    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        # name = k[7:] # remove module.
        name = k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)

    valid_provider = Provider_valid(cfg)
    val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1)

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

    model.eval()
    loss_all = []
    f_txt = open(os.path.join(out_affs, 'scores.txt'), 'w')
    print('the number of sub-volume:', len(valid_provider))
    losses_valid = []
    t1 = time.time()
    # valid_provider.reset_output(default_c=12)
    pbar = tqdm(total=len(valid_provider))
    for k, data in enumerate(val_loader, 0):
        inputs, target, weightmap = data
        inputs = inputs.cuda()
        target = target.cuda()
        weightmap = weightmap.cuda()
        if args.if_embedding:
            with torch.no_grad():
                if 'ct_embedding_mse' in args.cfg:
                    _, _, _, _, embedding = model(inputs)
                else:
                    embedding = model(inputs)
            # tmp_loss = criterion(pred, target, weightmap)

            tmp_loss, pred = embedding_loss_norm_multi(embedding, target, weightmap, criterion, affs0_weight=cfg.TRAIN.affs0_weight,shift=cfg.DATA.shift_channels)

            # pred = inf_embedding_loss_norm5(embedding)
            tmp_loss = 0.0
            shift = 1
            pred[:, 1, :, :shift, :] = pred[:, 1, :, shift:shift*2, :]
            pred[:, 2, :, :, :shift] = pred[:, 2, :, :, shift:shift*2]
            pred[:, 0, :shift, :, :] = pred[:, 0, shift:shift*2, :, :]
            pred = F.relu(pred)
            entropy = -(pred * torch.log(pred + 1e-10)+ (1-pred)*torch.log(1-pred+ 1e-10))

        else:
            with torch.no_grad():
                pred = model(inputs)
            tmp_loss = criterion(pred, target, weightmap)
        # losses_valid.append(tmp_loss.item())
        losses_valid.append(tmp_loss)
        valid_provider.add_vol(np.squeeze(pred.data.cpu().numpy()))
        valid_provider.add_vol_emb(np.squeeze(embedding.data.cpu().numpy()))
        valid_provider.add_vol_entropy(np.squeeze(entropy.data.cpu().numpy()))
        pbar.update(1)
        print(valid_provider.out_embs.max())
    pbar.close()

    cost_time = time.time() - t1
    print('Inference time=%.6f' % cost_time)
    f_txt.write('Inference time=%.6f' % cost_time)
    f_txt.write('\n')
    epoch_loss = sum(losses_valid) / len(losses_valid)
    embeddings = valid_provider.out_embs
    entropy = valid_provider.out_entropy

    output_affs = valid_provider.get_results()
    gt_affs = valid_provider.get_gt_affs()
    gt_seg = valid_provider.get_gt_lb()
    raw_data = valid_provider.get_raw_data()
    valid_provider.reset_output()
    

    # save
    # print('save affs...')
    # print('the shape of affs:', output_affs.shape)
    # f = h5py.File(os.path.join(out_affs, 'affs.hdf'), 'w')
    # f.create_dataset('main', data=output_affs, dtype=np.float32, compression='gzip')
    # f.close()

    print('affinity shape:', output_affs.shape)


    if args.save_affs:
        print('save affs...')
        # print('the shape of affs:', output_affs.shape)
        f = h5py.File(os.path.join(out_affs, 'affs.hdf'), 'w')
        f.create_dataset('main', data=output_affs, dtype=np.float32, compression='gzip')
        f.close()

        print('save target groundtruth segmentation...')
        f = h5py.File(os.path.join(out_affs, 'gt.hdf'), 'w')
        f.create_dataset('main', data=gt_seg, dtype=np.float32, compression='gzip')
        f.close()

    # for waterz
    output_affs = output_affs[:3]

    print('segmentation...')
    fragments = watershed(output_affs, 'maxima_distance')
    sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
    # sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
    segmentation = list(waterz.agglomerate(output_affs, [0.50],
                                        fragments=fragments,
                                        scoring_function=sf,
                                        discretize_queue=256))[0]
    segmentation = relabel(segmentation).astype(np.uint64)
    print('the max id = %d' % np.max(segmentation))
    f = h5py.File(os.path.join(out_affs, 'seg_waterz.hdf'), 'w')
    f.create_dataset('main', data=segmentation, dtype=segmentation.dtype, compression='gzip')
    f.close()

    arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
    voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
    voi_sum = voi_split + voi_merge
    print('waterz: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
        (voi_split, voi_merge, voi_sum, arand))
    f_txt.write('waterz: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
        (voi_split, voi_merge, voi_sum, arand))
    f_txt.write('\n')

    # segmentation = mc_baseline(output_affs,fragments=None,thres=args.lmc_thres)
    # segmentation = relabel(segmentation).astype(np.uint64)
    # print('the max id = %d' % np.max(segmentation))
    # f = h5py.File(os.path.join(out_affs, 'seg_lmc.hdf'), 'w')
    # f.create_dataset('main', data=segmentation, dtype=segmentation.dtype, compression='gzip')
    # f.close()

    # arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
    # voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
    # voi_sum = voi_split + voi_merge
    # print('LMC: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
    #     (voi_split, voi_merge, voi_sum, arand))
    # f_txt.write('LMC: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
    #     (voi_split, voi_merge, voi_sum, arand))
    # f_txt.write('\n')

    # compute MSE
    if args.pixel_metric:
        print('MSE...')
        output_affs_prop = output_affs.copy()
        whole_mse = np.sum(np.square(output_affs - gt_affs)) / np.size(gt_affs)
        print('BCE...')
        output_affs = np.clip(output_affs, 0.000001, 0.999999)
        bce = -(gt_affs * np.log(output_affs) + (1 - gt_affs) * np.log(1 - output_affs))
        whole_bce = np.sum(bce) / np.size(gt_affs)
        output_affs[output_affs <= 0.5] = 0
        output_affs[output_affs > 0.5] = 1
        print('F1...')
        whole_arand = 1 - f1_score(gt_affs.astype(np.uint8).flatten(), output_affs.astype(np.uint8).flatten())
        # whole_arand = 0.0
        # new
        print('F1 boundary...')
        whole_arand_bound = f1_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs.astype(np.uint8).flatten())
        # whole_arand_bound = 0.0
        print('mAP...')
        whole_map = average_precision_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs_prop.flatten())
        # whole_map = 0.0
        print('AUC...')
        whole_auc = roc_auc_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs_prop.flatten())
        # whole_auc = 0.0
        ###################################################
        malis = 0.0
        ###################################################
        print('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, ARAND-loss=%.6f, F1-bound=%.6f, mAP=%.6f, auc=%.6f, malis-loss=%.6f' % \
            (args.model_id, epoch_loss, whole_mse, whole_bce, whole_arand, whole_arand_bound, whole_map, whole_auc, malis))
        f_txt.write('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, ARAND-loss=%.6f, F1-bound=%.6f, mAP=%.6f, auc=%.6f, malis-loss=%.6f' % \
                    (args.model_id, epoch_loss, whole_mse, whole_bce, whole_arand, whole_arand_bound, whole_map, whole_auc, malis))
        f_txt.write('\n')
    else:
        output_affs_prop = output_affs
    f_txt.close()

    # show
    if args.show:
        print('show affs...')
        output_affs_prop = (output_affs_prop * 255).astype(np.uint8)
        gt_affs = (gt_affs * 255).astype(np.uint8)
        for i in range(output_affs_prop.shape[1]):
            cat1 = np.concatenate([output_affs_prop[0,i], output_affs_prop[1,i], output_affs_prop[2,i]], axis=1)
            cv2.imwrite(os.path.join(affs_img_path, str(i).zfill(4)+'.png'), cat1)
        for i in range(gt_affs.shape[1]):
            cat1 = np.concatenate([gt_affs[0,i], gt_affs[1,i], gt_affs[2,i]], axis=1)
            cv2.imwrite(os.path.join(affs_img_path, str(i).zfill(4)+'_gt.png'), cat1)
        for i in range(raw_data.shape[0]):
            cv2.imwrite(os.path.join(affs_img_path, str(i).zfill(4)+'_raw.png'), raw_data[i])
        
        for i in range(embeddings.shape[1]):
            emb = embeddings[:,i]
            emb_pca = embedding_pca(emb)
            emb_pca = np.transpose(emb_pca, (1,2,0))
            cv2.imwrite(os.path.join(affs_img_path, str(i).zfill(4)+'_emb.png'), emb_pca)
        for i in range(entropy.shape[1]):
            cv2.imwrite(os.path.join(affs_img_path, str(i).zfill(4)+'_entropy.png'), entropy[0,i])
            io.imsave(os.path.join(affs_img_path, str(i).zfill(4)+'_entropy0_2.png'), entropy[0,i])
            io.imsave(os.path.join(affs_img_path, str(i).zfill(4)+'_entropy1_2.png'), entropy[1,i])
            io.imsave(os.path.join(affs_img_path, str(i).zfill(4)+'_entropy2_2.png'), entropy[2,i])
            
    print('Done')
