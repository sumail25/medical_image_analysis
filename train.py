import argparse
import logging
import torch
import matplotlib.pyplot as plt

from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch import autograd, optim

from dataset import *
from metrics import *
from plot import loss_plot
from plot import metrics_plot

from UNet import Unet,resnet34_unet
#from attention_unet import AttU_Net
#from channel_unet import myChannelUnet
#from r2unet import R2U_Net
#from segnet import SegNet
#from unetpp import NestedUNet
#from fcn import get_fcn8s