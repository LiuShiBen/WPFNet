"""
 > Script for testing .pth models  
    * set model_name ('funiegan'/'ugan') and  model path
    * set data_dir (input) and sample_dir (output) 
"""
# py libs
import os
import time
import argparse
import numpy as np
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists
# pytorch libs
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms

from nets import Trans

def test(opt):
    '''  ------tesing pipeline------  '''
    assert exists(opt.model_path), "model not found"
    os.makedirs(opt.results_dir, exist_ok=True)
    is_cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor

    # model arch

    model = Trans.WPFNet()
    ## load weights
    model.load_state_dict(torch.load(opt.model_path))

    if is_cuda: model.cuda()
    print("Loaded model from %s" % (opt.model_path))

    ## data pipeline
    img_width, img_height, channels = 256, 256, 3
    transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]
    transform = transforms.Compose(transforms_)

    ## testing loop
    times = []
    test_files = sorted(glob(join(opt.data_dir, "*.*")))
    for path in test_files:
        inp_img = transform(Image.open(path))
        inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)
        # generate enhanced image
        s = time.time()
        gen_img, ll = model(inp_img)
        times.append(time.time() - s)
        save_image(gen_img[0], join(opt.sample_dir, basename(path)), normalize=True)
        #save_image(ll[0].data, join(opt.Ram_dir, basename(path)), normalize=True)
        # save_image(I_RGB, join(opt.RGB, basename(path)), normalize=True)
        print("Tested: %s" % path)


if __name__ == '__main__':
    ## options
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/test/", help="path of test images")
    parser.add_argument("--results_dir", type=str, default="./data/output/", help="path to save generated image")
    #parser.add_argument("--Ram_dir", type=str, default="./data/RAM/", help="path to save generated image")
    parser.add_argument("--model_path", type=str, default="checkpoint/generator_600.pth")
    opt = parser.parse_args()
    test(opt)

