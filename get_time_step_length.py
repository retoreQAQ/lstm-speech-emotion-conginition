import sys
import torch
import kaldiio

import os


def get_max(ark_path):
    feats_dict = kaldiio.load_ark(ark_path)
    feats_list = [[torch.tensor(feat), uttid] for uttid, feat in feats_dict]
    length_max = 0
    for feat_label in feats_list:
        length = feat_label[0].shape[0]
        d_num = feat_label[0].shape[1]
        if length > length_max: 
            length_max = length
    return length_max, d_num
# def check(ark_path):
#     error = []
#     feats_dict = kaldiio.load_ark(ark_path)
#     name_list = [uttid for uttid, feat in feats_dict]
#     for name in name_list:
#         n1 = int(name.split('-')[1])
#         n2 = int(name.split('-')[2])
#         n3 = int(name.split('-')[3])
#         if n1<4 or n2<4 or n3<4:
#             error.append(name)
#     return error

if __name__ == '__main__':
    path = os.getcwd()

    train_mfcc = './feat-for-type/train/mfcc_train_39.ark'
    val_mfcc = './feat-for-type/val/mfcc_val_39.ark'
    train_spec = './feat-for-type/train/spec_train.ark'
    val_spec = './feat-for-type/val/spec_val.ark'
    train_fbank = './feat-for-type/train/fbank_train.ark'
    val_fbank = './feat-for-type/val/fbank_val.ark'
    train_plp = './feat-for-type/train/plp_train_39.ark'
    val_plp = './feat-for-type/val/plp_val_39.ark'
    
    argument = None
    
    if argument == 'mfcc' or argument is None:
        l_train = get_max(train_mfcc)
        l_val = get_max(val_mfcc)
        max, n_features = l_train if l_train > l_val else l_val
        print(f'mfcc: seq_length is {max}, n_features is {n_features}.')
    if argument == 'spec' or argument is None:
        l_train = get_max(train_spec)
        l_val = get_max(val_spec)
        max, n_features = l_train if l_train > l_val else l_val
        print(f'spec: seq_length is {max}, n_features is {n_features}.')
    if argument == 'fbank' or argument is None:
        l_train = get_max(train_fbank)
        l_val = get_max(val_fbank)
        max, n_features = l_train if l_train > l_val else l_val
        print(f'fbank: seq_length is {max}, n_features is {n_features}.')
    if argument == 'plp' or argument is None:
        l_train = get_max(train_plp)
        l_val = get_max(val_plp)
        max, n_features = l_train if l_train > l_val else l_val
        print(f'plp: seq_length is {max}, n_features is {n_features}.')