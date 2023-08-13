import os
import cv2
import h5py
import waterz
import argparse
import tifffile
import numpy as np
from utils.fragment import watershed, randomlabel
from utils.fragment import relabel, remove_small
from utils.show import draw_fragments_3d
from scipy.ndimage import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--in_path', type=str, default='', help='path to config file')
    parser.add_argument('-id', '--model_id', type=int, default=51000)
    parser.add_argument('-m', '--mode', type=str, default='isbi')
    parser.add_argument('-sm', '--seg_mode', type=str, default='waterz')
    parser.add_argument('-wm', '--waterz_mode', type=str, default='')
    parser.add_argument('-cf', '--custom_fragments', action='store_false', default=True)
    parser.add_argument('-mf', '--mask_fragments', action='store_true', default=False)
    parser.add_argument('-dq', '--discrete_queue', action='store_false', default=True)
    parser.add_argument('-sf', '--score_func', type=str, default='median_aff_histograms')
    parser.add_argument('-rs', '--remove_small', type=float, default=None)
    parser.add_argument('-sw', '--show', action='store_true', default=False)
    parser.add_argument('-st', '--start_th', type=float, default=0.1)
    parser.add_argument('-et', '--end_th', type=float, default=0.9)
    parser.add_argument('-s', '--stride', type=float, default=0.1)
    args = parser.parse_args()

    trained_model = args.in_path
    out_path = os.path.join('../inference', trained_model, args.mode)
    img_folder = 'affs_'+str(args.model_id)
    out_affs = os.path.join(out_path, img_folder)
    print('out_path: ' + out_affs)
    seg_img_path = os.path.join(out_affs, 'seg_img')
    if not os.path.exists(seg_img_path):
        os.makedirs(seg_img_path)

    # load affs
    f = h5py.File(os.path.join(out_affs, 'affs.hdf'), 'r')
    affs = f['main'][:]
    f.close()

    if args.mode == 'isbi':
        print('isbi')
        f = h5py.File('../data/snemi3d/isbi_labels.h5', 'r')
        test_label = f['main'][:]
        f.close()
        test_label = test_label[80:]
    elif args.mode == 'isbi_test':
        print('isbi_test')
        f = h5py.File('../data/snemi3d/isbi_test_labels.h5', 'r')
        test_label = f['main'][:]
        f.close()
    elif args.mode == 'ac4':
        print('ac4')
        f = h5py.File('../data/ac3_ac4/AC4_labels.h5', 'r')
        test_label = f['main'][:]
        f.close()
        test_label = test_label[80:]
    elif args.mode == 'ac3':
        print('ac3')
        f = h5py.File('../data/ac3_ac4/AC3_labels.h5', 'r')
        test_label = f['main'][:]
        f.close()
    else:
        raise NotImplementedError
    test_label = test_label.astype(np.uint32)

    thresholds = np.arange(args.start_th, args.end_th+args.stride, args.stride)
    thresholds = list(thresholds)
    print('thresholds:', thresholds)

    if args.seg_mode == 'waterz':
        affs = affs[:3]
        if args.waterz_mode == 'default':
            print('waterz default')
            seg = waterz.agglomerate(affs, thresholds, gt=test_label)
        elif args.waterz_mode == 'waterz_fragment':
            print('waterz fragment')
            fragments = watershed(affs, 'maxima_distance')
            seg = waterz.agglomerate(affs, thresholds, gt=test_label, fragments=fragments)
        elif args.waterz_mode == 'mala':
            print('waterz mala')
            fragments = watershed(affs, 'maxima_distance')
            sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
            seg = waterz.agglomerate(affs, thresholds, gt=test_label, fragments=fragments, scoring_function=sf, discretize_queue=256)
        else:
            print('waterz others')
            fragments = None
            if args.custom_fragments:
                if args.mask_fragments:
                    no_gt = test_label == 0
                    no_gt = binary_erosion(no_gt, iterations=1, border_value=True)
                    # no_gt = binary_dilation(no_gt, iterations=1, border_value=True)
                    for d in range(3):
                        affs[d][no_gt] = 0
                    fragments_mask = no_gt==False
                fragments = watershed(affs, 'maxima_distance')
                if args.mask_fragments:
                    fragments[fragments_mask==False] = 0
            discretize_queue = 0
            if args.discrete_queue:
                discretize_queue = 256
            scoring_function = {
                'median_aff': 'OneMinus<QuantileAffinity<RegionGraphType, 50, ScoreValue>>',
                'median_aff_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>',
                '85_aff': 'OneMinus<QuantileAffinity<RegionGraphType, 85, ScoreValue>>',
                '85_aff_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256>>',
                'max_10': 'OneMinus<MeanMaxKAffinity<RegionGraphType, 10, ScoreValue>>'}
            seg = waterz.agglomerate(affs, thresholds, gt=test_label,
                                    fragments=fragments,
                                    scoring_function=scoring_function[args.score_func],
                                    discretize_queue=discretize_queue)
        best_arand = 1000
        best_idx = 0
        f_txt = open(os.path.join(out_affs, 'seg_waterz.txt'), 'w')
        seg_results = []
        for idx, seg_metric in enumerate(seg):
            segmentation = seg_metric[0].astype(np.int32)
            metrics = seg_metric[1]
            seg_results.append(segmentation)
            print('threshold=%.2f, voi_split=%.6f, voi_merge=%.6f, rand_split=%.6f, rand_merge=%.6f' % \
                (thresholds[idx], metrics['V_Info_split'], metrics['V_Info_merge'], metrics['V_Rand_split'], metrics['V_Rand_merge']))
            arand = adapted_rand_ref(test_label, segmentation, ignore_labels=(0))[0]
            voi_split, voi_merge = voi_ref(test_label, segmentation, ignore_labels=(0))
            voi_sum = voi_split + voi_merge
            print('threshold=%.2f, voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
                (thresholds[idx], voi_split, voi_merge, voi_sum, arand))
            f_txt.write('threshold=%.2f, voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
                (thresholds[idx], voi_split, voi_merge, voi_sum, arand))
            f_txt.write('\n')
            if voi_sum < best_arand:
                best_arand = voi_sum
                best_idx = idx
        print('Best threshold=%.2f, Best voi=%.6f' % (thresholds[best_idx], best_arand))
        segmentation = seg_results[best_idx]
    elif args.seg_mode == 'lmc':
        from utils.lmc import mc_baseline
        print('LMC...')
        affs = affs[:3]
        segmentation = mc_baseline(affs)
        segmentation = segmentation.astype(np.int32)
        f_txt = open(os.path.join(out_affs, 'seg_lmc.txt'), 'w')
    elif args.seg_mode == 'lmc_multi':
        from utils.lmc import multicut_multi
        print('LMC multi...')
        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                    [-2, 0, 0], [0, -3, 0], [0, 0, -3],
                    [-3, 0, 0], [0, -9, 0], [0, 0, -9],
                    [-4, 0, 0], [0, -27, 0], [0, 0, -27]]
        segmentation = multicut_multi(affs, offsets=offsets)
        segmentation = segmentation.astype(np.int32)
        f_txt = open(os.path.join(out_affs, 'seg_lmc_multi.txt'), 'w')
    elif args.seg_mode == 'mc':
        print("Multicut segmentation ...")
        from utils.mc_baselines import compute_mc_superpixels
        affs = affs[:3]
        segmentation = compute_mc_superpixels(1.0 - affs, n_threads=8)
        segmentation = segmentation.astype(np.int32)
        f_txt = open(os.path.join(out_affs, 'seg_mc.txt'), 'w')
    elif args.seg_mode == 'mutex':
        print("Mutex segmentation ...")
        from elf.segmentation.mutex_watershed import mutex_watershed
        shift = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                # direct 3d nhood for attractive edges
                [-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1],
                # indirect 3d nhood for dam edges
                [0, -9, 0], [0, 0, -9],
                # long range direct hood
                [0, -9, -9], [0, 9, -9], [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4],
                # inplane diagonal dam edges
                [0, -27, 0], [0, 0, -27]]
        strides = np.array([1, 10, 10])
        segmentation = mutex_watershed(1 - affs, shift, strides, randomize_strides=False)
        f_txt = open(os.path.join(out_affs, 'seg_mutex.txt'), 'w')
    else:
        raise NotImplementedError

    if args.remove_small is not None:
        segmentation = remove_small(segmentation, thres=args.remove_small)
    # segmentation = randomlabel(segmentation)
    segmentation = relabel(segmentation)
    print('The number of ids:', np.max(segmentation))
    segmentation = segmentation.astype(np.uint16)
    arand = adapted_rand_ref(test_label, segmentation, ignore_labels=(0))[0]
    voi_split, voi_merge = voi_ref(test_label, segmentation, ignore_labels=(0))
    voi_sum = voi_split + voi_merge
    print('voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
        (voi_split, voi_merge, voi_sum, arand))
    f_txt.write('voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
        (voi_split, voi_merge, voi_sum, arand))
    f_txt.write('\n')
    f_txt.close()

    f = h5py.File(os.path.join(out_affs, 'seg.hdf'), 'w')
    f.create_dataset('main', data=segmentation, dtype=np.uint16, compression='gzip')
    f.close()
    # tifffile.imwrite(os.path.join(out_affs, 'seg.tif'), segmentation)

    if args.show:
        print('show seg...')
        # segmentation[test_label==0] = 0
        color_seg = draw_fragments_3d(segmentation)
        color_gt = draw_fragments_3d(test_label)
        for i in range(color_seg.shape[0]):
            im_cat = np.concatenate([color_seg[i], color_gt[i]], axis=1)
            cv2.imwrite(os.path.join(seg_img_path, str(i).zfill(4)+'.png'), im_cat)
    
    print('Done')
