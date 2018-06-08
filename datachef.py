import warnings
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', FutureWarning)


import numpy as np
from collections import defaultdict
import random, cPickle, sys, os, csv, json, operator, cv2, chainercv, copy, h5py
from chainer.dataset import dataset_mixin
import chainer


class ReID10D(dataset_mixin.DatasetMixin):
    def __init__(self, args, image_list_path, image_size=(512, 512)):
        self.pairs = []
        F = open(image_list_path).readlines()
        for f in F:
            image_path, label = f.split()
            image_path = '%s/%s' % (args.dataset_folder, image_path)
            self.pairs.append((np.array(label, dtype=np.int32), image_path))        
        self.image_size = image_size
        self.meanchannel = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    def __len__(self):
        return len(self.pairs)

    def get_example(self, i):
        # Read image (BGR)
        label, image_path = self.pairs[i]        
        input_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # Resize image
        fy = float(self.image_size[0]) / float(input_image.shape[0])  # Height
        fx = float(self.image_size[1]) / float(input_image.shape[1])  # Width
        input_image = cv2.resize(input_image, None, None, fx=fx, fy=fy,
            interpolation=cv2.INTER_LINEAR).astype(np.float32)
        # Mean-channel subtraction
        input_image -= self.meanchannel
        # WxHx3 to 3xWxH
        input_image = np.transpose(input_image, [2, 0, 1])
        return input_image, label


def Report(history, report_interval, iterk=None, T=None,  split='train'):
    # Colors for print in terminal ;)
    c1 = "\033[40;1;31m"
    c2 = "\033[40;1;34m"
    c3 = "\033[40;1;32m"
    c4 = "\033[40;1;36m"
    c0 = "\033[0m"
    vs_domain, ks_domain = [], []
    cs_domain = [c1,c2,c3,c4]
    for prefix in history.keys():
        for k in history[prefix].keys():        
            vs_domain.append(np.asarray(history[prefix][k][-report_interval:]))
            ks_domain.append(k)
    # Report format
    report = []        
    report.append('%s after %.2f/%d hours/iters:' % (split, T/3600.0, iterk))    
    # Loss and average precision
    for k_domain, c_domain, v_domain in zip(ks_domain,cs_domain,vs_domain):
        report.append('%s:%s%.4f%s'%(k_domain,c_domain,np.nanmean(v_domain),c0))
    print '  '.join(report)
