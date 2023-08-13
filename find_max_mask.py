'''
Descripttion: 
version: 0.0
Author: Wei Huang
Date: 2021-11-28 15:47:18
'''
import os
import h5py
# import zarr
import argparse
import numpy as np

def matched_score(pred_mask, label_mask):
    overlap = pred_mask * label_mask
    mask = pred_mask + label_mask
    mask[mask!=0] = 1
    score = np.sum(overlap) / np.sum(mask)
    return score

def find_id(pred, mask, size=10000):
    split_ids = []
    pred = pred * mask
    ids, count = np.unique(pred, return_counts=True)
    id_dict = {}
    for k,v in zip(ids, count):
        if k == 0:
            continue
        if v > size:
            split_ids.append(k)
        id_dict.setdefault(k, v)
    sorted_results = sorted(id_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    new_id = sorted_results[0][0]
    if new_id == 0:
        new_id = sorted_results[1][0]
    pred_mask = np.zeros_like(pred)
    pred_mask[pred==new_id] = 1
    score = matched_score(pred_mask, mask)
    return new_id, score, split_ids, pred_mask

def match_mask(pred, label, f_txt, gt_ids=None, remove_list=[0], min_size=100000, mask_pred=False):
    '''
    pred: [D, H, W]
    gt: [D, H, W]
    num: the deised numbers used to display
    '''
    if mask_pred:
        pred[label==0] = 0
    if gt_ids is None:
        print('gen gt ids...')
        ids, count = np.unique(label, return_counts=True)
        id_dict = {}
        for k,v in zip(ids, count):  # id and corresponding pixels
            if k in remove_list:
                continue
            if v < min_size:
                continue
            id_dict.setdefault(k, v)
        sorted_results = sorted(id_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True) #decrease by number of pixels of each IDs
        print('the number of ids = %d' % len(sorted_results))
        gt_ids = []
        for k, v in sorted_results:
            gt_ids.append(k)

    for k in gt_ids:
        mask = np.zeros_like(label)
        mask[label==k] = 1
        matched_id, score, overlap_ids, _ = find_id(pred, mask)

        print('gt_id=%d, matched_id=%d, score=%.6f, overlap_ids=%d' % (k, matched_id, score, len(overlap_ids)))
        f_txt.write('gt_id=%d, matched_id=%d, score=%.6f, overlap_ids=%d' % (k, matched_id, score, len(overlap_ids)))
        f_txt.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--model_name', type=str, default='2021-12-25--13-58-54_seg_CremiA_dics_mala_o3_emb')
    parser.add_argument('-id', '--model_id', type=int, default=197000)
    parser.add_argument('-m', '--mode', type=str, default='cremiA')  # cremiA,fib2
    parser.add_argument('-nlmc', '--not_lmc', action='store_true', default=False)
    args = parser.parse_args()

    trained_model = args.model_name
    out_path = os.path.join('../inference', trained_model, args.mode)
    img_folder = 'affs_'+str(args.model_id)
    out_affs = os.path.join(out_path, img_folder)
    # print('out_path: ' + out_affs)

    print('Load labels...',args.mode)
    label_file = os.path.join('../inference_gt/', args.mode+'.hdf')
    print('label file: %s' % label_file)
    f = h5py.File(label_file, 'r')
    gt = f['main'][:]
    f.close()

    print('Load segmentation results...')

    seg_file = os.path.join(out_affs, 'seg_waterz.hdf')
    f = h5py.File(seg_file, 'r')
    print('seg file: %s' % seg_file)
    seg = f['main'][:]
    f.close()
    f_txt = open(os.path.join(out_affs, 'record_matched_id_waterz.txt'), 'w')
    # match_mask(seg, gt, f_txt, gt_ids=gt_ids)
    match_mask(seg, gt, f_txt, gt_ids=None)
    f_txt.close()
    print('Done')

    if args.not_lmc:
        print('Done')

    else:
        seg_file = os.path.join(out_affs, 'seg_lmc.hdf')
        f = h5py.File(seg_file, 'r')
        print('seg file: %s' % seg_file)
        seg = f['main'][:]
        f.close()
        f_txt = open(os.path.join(out_affs, 'record_matched_id_lmc.txt'), 'w')
        # match_mask(seg, gt, f_txt, gt_ids=gt_ids)
        match_mask(seg, gt, f_txt, gt_ids=None)
        f_txt.close()
        print('Done')

    

    # id_file = os.path.join('../data/superset', label_name+'_id.txt')
    # print('id file: %s' % id_file)
    # f_txt = open(id_file, 'r')
    # content = [x[:-1] for x in f_txt.readlines()]
    # f_txt.close()
    # gt_ids = []
    # for c in content:
    #     gt_ids.append(int(c.split(' ')[0]))


