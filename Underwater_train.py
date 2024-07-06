"""
 > Training pipeline for FUnIE-GAN (paired) model
   * Paper: arxiv.org/pdf/1903.09766.pdf
 > Maintainer: https://github.com/xahidbuffon
"""
# py libs
import os
import sys
import yaml
import argparse
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from os.path import join, exists
from glob import glob
from ntpath import basename
from nets.Dwt import DWTForward
# local libs
from nets.commons import Weights_Normal, VGG19_PercepLoss
from nets.Trans import WPFNet
#from nets.MTUNet import MTUNet
from utils.data_utils import GetTrainingPairs, GetValImage
from CR import ContrastLoss, VGG19_PercepLoss
from torchvision.transforms import Resize

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class L_RGBloss(nn.Module):
    def __init__(self):
        super(L_RGBloss, self).__init__()
    def forward(self,x):

        mean_rgb = torch.mean(x,[2,3],keepdim = True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        k = torch.mean(k)
        return k


class Laplace(nn.Module):
    def __init__(self):
        super(Laplace, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
        nn.init.constant_(self.conv1.weight, 1)
        nn.init.constant_(self.conv1.weight[0, 0, 1, 1], -8)
        nn.init.constant_(self.conv1.weight[0, 1, 1, 1], -8)
        nn.init.constant_(self.conv1.weight[0, 2, 1, 1], -8)

    def forward(self, x1):
        edge_map = self.conv1(x1)
        return edge_map

def get_scheduler(optimizer):

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch) / float(600)
        #print(lr_l)
        return lr_l

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    #print("...", scheduler)
    return scheduler

# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    #print('learning rate = %.7f' % lr)

## get configs and training options
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--dataset_path", type=str, default= "./data/UIEB/", help="path of train images")
parser.add_argument("--img_width", type=int, default=256, help="width of image")
parser.add_argument("--img_height", type=int, default=256, help="height of image")
parser.add_argument("--val_interval", type=int, default=100, help="width of image")
parser.add_argument("--ckpt_interval", type=int, default=50, help="Every 10 epoch of saving model checkpoints")
parser.add_argument("--num_epochs", type=int, default=601, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of 1st order momentum")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of 2nd order momentum")
parser.add_argument("--n_critic", type=int, default=5, help="training steps for D per iter w.r.t G")

args = parser.parse_args()

## training params
epoch = args.epoch
num_epochs = args.num_epochs
batch_size =  args.batch_size
lr_rate, lr_b1, lr_b2 = args.lr, args.b1, args.b2
num_critic = args.n_critic

model_v = "WPFNet"
dataset_path = args.dataset_path
img_width = args.img_width
img_height = args.img_height
val_interval = args.val_interval
ckpt_interval = args.ckpt_interval


## create dir for model and validation data
checkpoint_dir = "checkpoints/%s/" % (model_v)
os.makedirs(checkpoint_dir, exist_ok=True)

""" WPFNet specifics: loss functions """

L2_G = torch.nn.MSELoss() #l2 loss
L1_G  = torch.nn.SmoothL1Loss() # l1_loss
L_vgg = VGG19_PercepLoss() # content loss (vgg)

'''......Discrete wavelet transform......'''
dwt1 = DWTForward(J=1, wave='db1', mode='zero')
dwt2 = DWTForward(J=1, wave='db1', mode='zero')
dwt3 = DWTForward(J=1, wave='db1', mode='zero')
dwt4 = DWTForward(J=1, wave='db1', mode='zero')

# Initialize network
Generator = WPFNet()
Lap = Laplace()
# see if cuda is available
if torch.cuda.is_available():
    Generator = Generator.cuda()
    dwt1 = dwt1.cuda()
    dwt2 = dwt2.cuda()
    dwt3 = dwt3.cuda()
    dwt4 = dwt4.cuda()
    L2_G = L2_G.cuda()
    L1_G = L1_G.cuda()
    L_vgg = L_vgg.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

# Initialize weights or load pretrained models
if args.epoch == 0:
    Generator.apply(Weights_Normal)

else:
    Generator.load_state_dict(torch.load("checkpoints/%s/generator_%d.pth" % (model_v, args.epoch)))
    print("Loaded model from epoch %d" %(epoch))

# Optimizers
optimizer_G = torch.optim.Adam(Generator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))
#scheduler = get_scheduler(optimizer_G)

## Data pipeline
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    GetTrainingPairs(dataset_path, transforms_=transforms_),
    batch_size = batch_size,
    shuffle = True,
    num_workers = 8,
)

# Defining Resize Class Objects
resize_128 = Resize([img_height //2, img_width//2])
resize_64 = Resize([img_height //4, img_width//4])
resize_32 = Resize([img_height //8, img_width//8])
resize_16 = Resize([img_height //16, img_width//16])

Loss = []
Psnr = []
Ssim = []

## Training pipeline
for epoch in range(epoch, num_epochs):
    for i, batch in enumerate(dataloader):
        # Model inputs
        imgs_distorted = Variable(batch["A"].type(Tensor))
        gt = Variable(batch["B"].type(Tensor))
        gll_128, _ = dwt1(gt)
        gll_64, _ = dwt2(gll_128)
        gll_32, _ = dwt3(gll_64)
        gll_16, _ = dwt4(gll_32)

        gt_128 = resize_128(gt)
        gt_64 = resize_64(gt)
        gt_32 = resize_32(gt)
        gt_16 = resize_16(gt)
        optimizer_G.zero_grad()
        out, ll = Generator(imgs_distorted)
        fake, fake_128, fake_64, fake_32, fake_16 = out[0], out[1], out[2], out[3], out[4]
        ll_128, ll_64, ll_32, ll_16 = ll[0], ll[1],ll[2], ll[3]
        #print(fake.shape, fake_128.shape, fake_64.shape, fake_32.shape)

        # Total loss
        loss_rec = 15 * (L1_G(fake, gt) + 1/4 * L1_G(fake_128, gt_128) + 1/8 * L1_G(fake_64, gt_64) + 1/16 * L1_G(fake_32, gt_32) + 1/32 * L1_G(fake_16, gt_16)) # reconstruction loss
        loss_fre = L1_G(ll_128, gll_128) + 1/4 * L1_G(ll_64, gll_64) + 1/8 * L1_G(ll_32, gll_32) + 1/16 * L1_G(ll_16, gll_16)  #frequency loss
        loss_vgg = 3 * (L_vgg(fake, gt))  # content loss

        loss = loss_rec + loss_vgg + loss_fre
        if(loss <= 10):
            Loss.append(loss.item())
        loss.backward()
        optimizer_G.step()

        if not i % 10:
            sys.stdout.write(
                "\r[Epoch %d/%d: batch %d/%d] [Loss: %.3f, loss_sim: %.3f, loss_vgg: %.3f, loss_frequency: %.3f]"
                % (
                    epoch, num_epochs, i, len(dataloader),
                    loss.item(), loss_rec.item(), loss_vgg.item(), loss_fre.item())
            )
            # If at sample interval save image

    #update_learning_rate(scheduler, optimizer_G)
    if (epoch % ckpt_interval == 0):
        torch.save(Generator.state_dict(), "checkpoints/%s/generator_%d.pth" % (model_v, epoch))

