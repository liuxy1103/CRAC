import numpy as np
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def embedding2affinity_hop(embedding, target, weightmap, criterion, affs0_weight=1, shift=1, fill=True):
    '''
    [[-1,0,0],
        [0,-1,0],
        [0,0,-1],
        [-2,0,0],
        [0,-3,0],
        [0,0,-3],
        [-3,0,0],
        [0,-9,0],
        [0,0,-9],
        [-4,0,0],
        [0,-27,0],
        [0,0,-27]]
    '''
    embedding = F.normalize(embedding, p=2, dim=1)
    B, C, D, H, W = embedding.shape
    affs = torch.zeros_like(target)
    loss = torch.tensor(0).cuda()
    affs_tmp_z = torch.sum(embedding[:, :, 1:, 0:, 0:]*embedding[:, :, :D-1, :H-0, :W-0], dim=1, keepdim=True)
    affs_tmp_y = torch.sum(embedding[:, :, 0:, 1:, 0:]*embedding[:, :, :D-0, :H-1, :W-0], dim=1, keepdim=True)
    affs_tmp_x = torch.sum(embedding[:, :, 0:, 0:, 1:]*embedding[:, :, :D-0, :H-0, :W-1], dim=1, keepdim=True)
    for i in range(len(shift)):
        z_sf, y_sf, x_sf = shift[i]
        # print(shift[i])
        # if z_sf<=0 and y_sf<=0 and x_sf<=0:
        if i <=2:  
            affs[:, i:i+1, -z_sf:, -y_sf:, -x_sf:] = affs_tmp
            loss_tmp = criterion(affs_tmp, target[:, i:i+1, -z_sf:, -y_sf:, -x_sf:], weightmap[:, i:i+1, -z_sf:, -y_sf:, -x_sf:])
        elif i>2 and z_sf<0:
            affs[:, i:i+1, -z_sf:, -y_sf:, -x_sf:] = 1- torch.relu(target[:, i:i+1, -z_sf:, -y_sf:, -x_sf:])
            loss_tmp = criterion(affs_tmp, target[:, i:i+1, -z_sf:, -y_sf:, -x_sf:], weightmap[:, i:i+1, -z_sf:, -y_sf:, -x_sf:])

        if i <= 2 :
            loss_tmp = loss_tmp*affs0_weight

        loss = loss+loss_tmp

    return loss, affs

def affinity_hop(aff, target, weightmap, criterion, affs0_weight=1, shift=1, fill=True):
    affs_tmp_z = aff[:,0]
    affs_tmp_y = aff[:,1] 
    affs_tmp_x = aff[:,2]
    pass


def gen_affs(map1, map2=None, dir=0, shift=1, padding=True, background=False):
    if dir == 0 and map2 is None:
        raise AttributeError('map2 is none')
    map1 = map1.astype(np.float32)
    h, w = map1.shape
    if dir == 0:
        map2 = map2.astype(np.float32)
    elif dir == 1:
        map2 = np.zeros_like(map1, dtype=np.float32)
        map2[shift:, :] = map1[:h-shift, :]
    elif dir == 2:
        map2 = np.zeros_like(map1, dtype=np.float32)
        map2[:, shift:] = map1[:, :w-shift]
    else:
        raise AttributeError('dir must be 0, 1 or 2')
    dif = map2 - map1
    out = dif.copy()
    out[dif == 0] = 1
    out[dif != 0] = 0
    if background:
        out[map1 == 0] = 0
        out[map2 == 0] = 0
    if padding:
        if dir == 1:
            # out[:shift, :] = (map1[:shift, :] > 0).astype(np.float32)
            out[:shift, :] = out[2*shift:shift:-1, :]
        if dir == 2:
            # out[:, :shift] = (map1[:, :shift] > 0).astype(np.float32)
            out[:, :shift] = out[:, 2*shift:shift:-1]
    else:
        if dir == 1:
            out[:shift, :] = 0
        if dir == 2:
            out[:, :shift] = 0
    return out

def gen_affs_mutex(map1, map2, shift=0, padding=True, background=False):
    assert len(shift) == 3, 'the len(shift) must be 3'
    h, w = map1.shape
    map1 = map1.astype(np.float32)
    map2 = map2.astype(np.float32)

    if shift[1] <= 0 and shift[2] <= 0:
        map1[-shift[1]:, -shift[2]:] = map1[:h+shift[1], :w+shift[2]]
    elif shift[1] <= 0 and shift[2] > 0:
        map1[-shift[1]:, :w-shift[2]] = map1[:h+shift[1], shift[2]:]
    elif shift[1] > 0 and shift[2] <= 0:
        map1[:h-shift[1], -shift[2]:] = map1[shift[1]:, :w+shift[2]]
    elif shift[1] > 0 and shift[2] > 0:
        map1[:h-shift[1], :w-shift[2]] = map1[shift[1]:, shift[2]:]
    else:
        pass

    dif = map1 - map2
    out = dif.copy()
    out[dif == 0] = 1
    out[dif != 0] = 0
    if background:
        out[map1 == 0] = 0
        out[map2 == 0] = 0
    if padding:
        if shift[1] < 0:
            out[:-shift[1], :] = out[-2*shift[1]:-shift[1]:-1, :]
        elif shift[1] > 0:
            out[h-shift[1]:, :] = out[h-shift[1]-2:h-2*shift[1]-2:-1, :]
        else:
            pass
        if shift[2] < 0:
            out[:, :-shift[2]] = out[:, -2*shift[2]:-shift[2]:-1]
        elif shift[2] > 0:
            out[:, w-shift[2]:] = out[:, w-shift[2]-2:w-2*shift[2]-2:-1]
        else:
            pass
    else:
        if shift[1] < 0:
            out[:-shift[1], :] = 0
        elif shift[1] > 0:
            out[h-shift[1]:, :] = 0
        else:
            pass
        if shift[2] < 0:
            out[:, :-shift[2]] = 0
        elif shift[2] > 0:
            out[:, w-shift[2]:] = 0
        else:
            pass
    return out

def gen_affs_3d_offset(labels, offsets=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]], padding=False, background=True):
    assert len(offsets) == 3, 'the length of offsets must be 3'
    offsets = np.asarray(offsets, dtype=np.int32)
    assert np.sum(offsets) == np.sum(np.diag(offsets)), 'offsets must be diagonal matrix'
    shape = labels.shape
    affs = np.zeros((3,)+shape, dtype=np.float32)
    offset_dict = {}
    offset_dict.setdefault(0, -offsets[0, 0])
    offset_dict.setdefault(1, -offsets[1, 1])
    offset_dict.setdefault(2, -offsets[2, 2])
    for k in range(labels.shape[0]):
        for i in range(3):
            if i == 0:
                if k >= offset_dict[i]:
                    affs[i, k] = gen_affs(labels[k-offset_dict[i]], labels[k], dir=i, shift=offset_dict[i], padding=padding, background=background)
            else:
                affs[i, k] = gen_affs(labels[k], labels[k], dir=i, shift=offset_dict[i], padding=padding, background=background)
    if padding:
        affs[0,0] = (labels[0]>0).astype(affs.dtype)
        affs[1,:,0] = (labels[:,0]>0).astype(affs.dtype)
        affs[2,:,:,0] = (labels[:,:,0]>0).astype(affs.dtype)
    return affs

def gen_affs_3d(labels, shift=1, padding=True, background=False):
    assert len(labels.shape) == 3, '3D input'
    out = []
    for i in range(labels.shape[0]):
        if i == 0:
            if padding:
                # affs0 = (labels[0] > 0).astype(np.float32)
                affs0 = gen_affs(labels[i], labels[i+1], dir=0, shift=shift, padding=padding, background=background)
            else:
                affs0 = np.zeros_like(labels[0], dtype=np.float32)
        else:
            affs0 = gen_affs(labels[i-1], labels[i], dir=0, shift=shift, padding=padding, background=background)
        affs1 = gen_affs(labels[i], None, dir=1, shift=shift, padding=padding, background=background)
        affs2 = gen_affs(labels[i], None, dir=2, shift=shift, padding=padding, background=background)
        affs = np.stack([affs0, affs1, affs2], axis=0)
        out.append(affs)
    out = np.asarray(out, dtype=np.float32)
    out = np.transpose(out, (1, 0, 2, 3))
    return out

def gen_affs_mutex_3d(labels, shift=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]], padding=True, background=False):
    affs = []
    for shift_k in shift:
        affs_k = []
        for i in range(labels.shape[0]):
            if shift_k[0] != 0:
                if i == 0:
                    if padding:
                        temp = gen_affs_mutex(labels[0], labels[1], shift=shift_k, padding=padding, background=background)
                    else:
                        temp = np.zeros_like(labels[0], dtype=np.float32)
                else:
                    temp = gen_affs_mutex(labels[i-1], labels[i], shift=shift_k, padding=padding, background=background)
            else:
                temp = gen_affs_mutex(labels[i], labels[i], shift=shift_k, padding=padding, background=background)
            affs_k.append(temp)
        affs.append(affs_k)
    affs = np.asarray(affs)
    return affs
