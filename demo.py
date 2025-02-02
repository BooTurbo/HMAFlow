import sys
sys.path.append('core')

import argparse
import os
import time
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from hmaflow import HMAFlow
from utils import flow_viz
from utils.utils import InputPadder


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0) 
    cv2.waitKey()  # 30


def flo_viz_save(flo, save_dir):
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    
    # flip flo from RGB to BGR
    cv2.imwrite(save_dir, flo[:, :, ::-1], [int(cv2.IMWRITE_PNG_COMPRESSION), 0]) 


def demo(args):
    model = torch.nn.DataParallel(HMAFlow(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            
            # iters=20
            _, flow_up = model(image1, image2, iters=32, test_mode=True)
            viz(image1, flow_up)
            
            ### Save flow_viz
            # im1_name = os.path.basename(imfile1)
            # seq_name = os.path.basename(args.path)
            # save_dir = os.path.join('demo-frames/test/flow/', seq_name, im1_name)
            # flo_viz_save(flow_up, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    
    args = parser.parse_args()

    demo(args)

