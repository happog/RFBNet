from __future__ import print_function
import sys
import os
from os.path import join
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
from data import AnnotationTransform, BaseTransform, VOC_300,VOC_512,COCO_300,COCO_512, COCO_mobile_300
import cv2
from layers.functions import Detect,PriorBox
from models.RFB_Net_E_vgg import build_net
from utils.nms.cpu_nms import cpu_nms
import mmcv
import time

import global_model as gm

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

def detect(data_dir, imgfile, reso=512, confidence=0.1, nms_thesh=0.4):
    """
    检测接口：输入样本数据集路径和待检测图像路径，利用yolov3模型进行检测
    :data_dir: 输入样本数据集目录
    :imgfile: 输入测试图像路径
    :reso: 设置预测分辨率，默认为608
    :confidence: 设置置信度阈值，默认为0.8
    :nms_thesh: 设置nms阈值，默认为0.4
    :return: 返回预测结果result, result[0]为列表[[x,y,w,h,score,class]; result[1]为备用的numpy矩阵
    """
    data_dir = os.path.abspath(data_dir)
    # load label names
    try:
        classes = load_classes(os.path.join(data_dir, 'ocr.names'))
    except IOError:
        print('detection: cannot read names file')
        return [],[]
    
    # model config
    numclass = len(classes) + 1
    cfg = COCO_512
    use_cuda = torch.cuda.is_available()
    trained_model = join(data_dir, 'logs/latest.pth')

    priorbox = PriorBox(cfg)
    with torch.no_grad():
        priors = priorbox.forward()
        if use_cuda:
            priors = priors.cuda()

    start_load = time.time()
    img = cv2.imread(imgfile)
    scale = torch.Tensor([img.shape[1], img.shape[0],
                        img.shape[1], img.shape[0]])
    
    # construct the model and load checkpoint
    if data_dir != gm.model_id:
        print("Loading network.....")
        net = build_net('test', reso, numclass)    # initialize detector
        state_dict = torch.load(trained_model, map_location='cpu')
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
        print("Network successfully loaded")
        gm.model_id = data_dir
        gm.model = net
    else:
        print('using the last loaded network')
        net = gm.model

    transform = BaseTransform(net.size, (123, 117, 104), (2, 0, 1))
    with torch.no_grad():
        x = transform(img).unsqueeze(0)
        if use_cuda:
            x = x.cuda()
            scale = scale.cuda()
    net.eval()
    if use_cuda:
        net = net.cuda()
        # cudnn.benchmark = True
    else:
        net = net.cpu()
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
    result = []
    for j in range(1, numclass):
        inds = np.where(scores[:, j] > confidence)[0]
        if inds is None:
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        # keep = nms_py(c_dets, nms_thesh)
        keep = cpu_nms(c_dets, nms_thesh)
        c_dets = c_dets[keep, :]
        c_bboxes=c_dets[:, :5]
        result.append(c_bboxes)

    # filter predict
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(result)

    # output predict
    predict = []
    for i in range(bboxes.shape[0]):
        predict.append([int(bboxes[i,0]),int(bboxes[i,1]),int(bboxes[i,2]-bboxes[i,0]+1),int(bboxes[i,3]-bboxes[i,1]+1),bboxes[i,-1],classes[int(labels[i])]])
    end = time.time()
    print('detection: predict finished in %2.2f sec with %d objects'%(end-start_load, len(predict)))

    # print(predict)
    return predict, result


if __name__ ==  '__main__':

    # data_dir = '/datastore2/shanghai_medical/'
    data_dir = './test_data/'
    img_list = 'test_data/list.txt'
    det='test_data/ocr_res/'

    # batch test
    imgs = load_classes(img_list)
    pred = lambda x: save_result(x, detect(data_dir, x)[1], out_file=det+x.split('/')[-1], namesfile=data_dir+'/ocr_en.names', score_thr=0)
    list(map(pred, imgs))

