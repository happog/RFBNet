from __future__ import print_function
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import VOCroot
from data import AnnotationTransform, BaseTransform, VOC_300,VOC_512,COCO_300,COCO_512, COCO_mobile_300
import cv2
import torch.utils.data as data
from layers.functions import Detect,PriorBox
from utils.nms.cpu_nms import cpu_nms
from utils.timer import Timer
import mmcv
import time

def nms_py(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)
    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    return keep

def load_classes(namesfile):
    # fp = open(namesfile, "r")
    fp = open(namesfile, "r", encoding='utf8')
    names = fp.read().split("\n")
    names = [x for x in names if len(x) > 0]
    return names

def save_result(img, result, out_file, namesfile, score_thr=0.8):
    class_names = load_classes(namesfile)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(result)
    img = mmcv.imread(img)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        bbox_color='red',
        text_color='blue',
        thickness=2,
        font_scale=0.8,
        show=False,
        out_file=out_file)

parser = argparse.ArgumentParser(description='Receptive Field Block Net')

parser.add_argument('-v', '--version', default='RFB_E_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default=512,
                    help='300 or 512 input size.')
parser.add_argument('-i', '--img', default='test_data/ocr_data/sample_1_1999.png',
                    help='VOC or COCO version')
parser.add_argument('-m', '--trained_model', default='logs/RFB_E_vgg_COCO_epoches_10.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
args = parser.parse_args()

if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net
    cfg = COCO_mobile_300
else:
    print('Unkown version!')
# cfg = VOC_300
cfg = COCO_512

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()
# numclass = 21
numclass = 15
start_load = time.time()
img = cv2.imread(args.img)
scale = torch.Tensor([img.shape[1], img.shape[0],
                     img.shape[1], img.shape[0]])
net = build_net('test', args.size, numclass)    # initialize detector

transform = BaseTransform(net.size, (123, 117, 104), (2, 0, 1))
with torch.no_grad():
    x = transform(img).unsqueeze(0)
    if args.cuda:
        x = x.cuda()
        scale = scale.cuda()
state_dict = torch.load(args.trained_model, map_location='cpu')
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:] # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)
net.eval()
if args.cuda:
    net = net.cuda()
    # cudnn.benchmark = True
else:
    net = net.cpu()
print('Finished loading model!')
# print(net)
start = time.time()
detector = Detect(numclass,0,cfg)
out = net(x)      # forward pass
boxes, scores = detector.forward(out,priors)
boxes = boxes[0]
scores=scores[0]
boxes *= scale
boxes = boxes.cpu().numpy()
scores = scores.cpu().numpy()

# scale each detection back up to the image
bboxes = []
for j in range(1, numclass):
    inds = np.where(scores[:, j] > 0.1)[0]      #conf > 0.6
    if inds is None:
        continue
    c_bboxes = boxes[inds]
    c_scores = scores[inds, j]
    c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
        np.float32, copy=False)
    # keep = nms_py(c_dets, 0.6)
    keep = cpu_nms(c_dets, 0.4)
    c_dets = c_dets[keep, :]
    c_bboxes=c_dets[:, :5]
    bboxes.append(c_bboxes)
end = time.time()
print('detection: load model in %2.2f sec and predict finished in %2.2f sec'%(start-start_load, end-start))

data_dir = './test_data/'
x = args.img
det='test_data/ocr_res/'
save_result(x, bboxes, out_file=det+x.split('/')[-1], namesfile=data_dir+'/ocr_en.names', score_thr=0)

