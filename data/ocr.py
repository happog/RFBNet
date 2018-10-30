"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import os
import pickle
from os.path import join
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from pypinyin import pinyin, lazy_pinyin, Style
from .label_map import label_map


def get_pinyin(names):
    piny = lazy_pinyin(names)
    piny = ''.join([x[0][0] for x in piny])
    return piny

def save_pinyin(namesfile, names):
    piny = list(map(get_pinyin, names))
    fp = open(namesfile, "w")
    [fp.write(x+'\n') for x in piny]
    return

def load_classes(namesfile):
    # fp = open(namesfile, "r")
    fp = open(namesfile, "r", encoding='utf8')
    names = fp.read().split("\n")
    names = [x for x in names if len(x) > 0]
    return names

class OCRDetection(data.Dataset):

    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_sets, preproc=None, target_transform=None,
                 dataset_name='OCR'):
        # generate list file
        ann_file = join(root, 'train.txt')
        img_prefix = join(root, 'images/')
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        os.system('ls %s/*.png >%s && ls %s/*.png >%s'%(img_prefix, ann_file, img_prefix, ann_file.replace('train','val')))
        # get img_ids and img_infos
        lines = open(ann_file).read().split('\n')
        lines = [x for x in lines if len(x) > 0] #get rid of the empty lines 
        lines = [x for x in lines if x[0] != '#']  
        lines = [x.rstrip().lstrip() for x in lines]
        self.img_ids = lines

        # get the mapping from original category ids to labels
        self.cat_ids = load_classes(ann_file.replace('train.txt','ocr.names'))
        save_pinyin(ann_file.replace('train.txt','ocr_en.names'), self.cat_ids)
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }

    def _parse_ann_info(self, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        
        labels = open(ann_info.replace("images","labels")[:-4]+".label",encoding="utf8").read().split('\n')
        labels = [x for x in labels if len(x) > 0]
        # w,h = [int(x) for x in labels[0].split(',')]
        # print(w,h)
        bbox = []
        for label in labels[1:]:
            label = label.split(',')
            cls = label[-1]
            # if cls == '项目金额':
            #     cls = '金额'
            # if cls in ['业务流水号','单价','金额','项目金额','年数值','月数值','日数值','条形码']:
            #     cls = '数值'
            # elif cls in ['项目规格','数量单位','等级','门诊大额支付','退休补充支付','残军补助支付','单位补充支付','本次医保范围内金额','累计医保范围内金额','年度门诊大额累计支付','本次支付后个人余额','自付一','超封顶金额','自付二','自费','起付金额']:
            #     cls = '条目'
            # elif cls in ['项目规格--表头','单价--表头','数量单位--表头','金额--表头','等级--表头','项目规格2--表头','单价2--表头','数量单位2--表头','金额2--表头','等级2--表头','基金支付--表头','个人账户支付--表头','个人支付金额--表头','收款单位--表头','收款人--表头','年--表头','月--表头','日--表头','发票号--表头','业务流水号--表头']:
            #     cls = '其他表头'
            cls = label_map['sh'].get(cls, '其他')
            if cls not in self.cat_ids:
                # print(cls+" is not in classes!")
                continue
            cls_id = self.cat2label[cls]
            x1, y1, w, h = [float(a) for a in label[0:4]]
            if w < 1 or h < 1:
                continue
            bbox.append([x1, y1, x1 + w - 1, y1 + h - 1, cls_id])
        ann = np.array(bbox)

        return ann

    def __getitem__(self, idx):
        img = cv2.imread(self.img_ids[idx], cv2.IMREAD_COLOR)
        target = self._parse_ann_info(self.img_ids[idx])


        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return img, target

    def __len__(self):
        return len(self.img_ids)

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.img_ids[index]
        return cv2.imread(img_id, cv2.IMREAD_COLOR)


    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        # to_tensor = transforms.ToTensor()
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

