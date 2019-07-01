"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
import sys
sys.path.append('.')
from trainer import DGNet_Trainer, to_gray
from utils import get_config
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import random
import os
import numpy as np
from torchvision import datasets, models, transforms
from PIL import Image
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('--output_folder', type=str, default="../Market/pytorch/off-gan_id1/", help="output image path")
parser.add_argument('--output_folder2', type=str, default="../Market/pytorch/off-gan_bg1/", help="output image path")
parser.add_argument('--input_folder', type=str, default="../Market/pytorch/train_all/", help="input image path")

parser.add_argument('--name', type=str, default="E0.5new_reid0.5_w30000", help="model name")
parser.add_argument('--which_epoch', default=100000, type=int, help='iteration')

parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--trainer', type=str, default='DGNet', help="DGNet")


opts = parser.parse_args()
opts.checkpoint_gen = "./outputs/%s/checkpoints/gen_00%06d.pt"%(opts.name, opts.which_epoch)
opts.checkpoint_id = "./outputs/%s/checkpoints/id_00%06d.pt"%(opts.name, opts.which_epoch)
opts.config = './outputs/%s/config.yaml'%opts.name

if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)
else:
    os.system('rm -rf %s/*'%opts.output_folder)

if not os.path.exists(opts.output_folder2):
    os.makedirs(opts.output_folder2)
else:
    os.system('rm -rf %s/*'%opts.output_folder2)

# Load experiment setting
config = get_config(opts.config)
# we use config
config['apex'] = False
opts.num_style = 1

# Setup model and data loader
if opts.trainer == 'DGNet':
    trainer = DGNet_Trainer(config)
else:
    sys.exit("Only support DGNet")

state_dict_gen = torch.load(opts.checkpoint_gen)
trainer.gen_a.load_state_dict(state_dict_gen['a'], strict=False)
trainer.gen_b = trainer.gen_a

state_dict_id = torch.load(opts.checkpoint_id)
trainer.id_a.load_state_dict(state_dict_id['a'])
trainer.id_b = trainer.id_a

trainer.cuda()
trainer.eval()
encode = trainer.gen_a.encode # encode function
style_encode = trainer.gen_a.encode # encode function
id_encode = trainer.id_a # encode function
decode = trainer.gen_a.decode # decode function

data_transforms = transforms.Compose([
        transforms.Resize(( config['crop_image_height'], config['crop_image_width']), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = datasets.ImageFolder(opts.input_folder, data_transforms)
dataloader_content = torch.utils.data.DataLoader(image_datasets, batch_size=opts.batchsize, shuffle=False, pin_memory=True, num_workers=1)
dataloader_structure = torch.utils.data.DataLoader(image_datasets, batch_size=opts.batchsize, shuffle=True, pin_memory=True, num_workers=1)
image_paths = image_datasets.imgs

######################################################################
# recover image
# -----------------
def recover(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = inp * 255.0
    inp = np.clip(inp, 0, 255)
    return inp

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

###########################################
# ID with different background (run 10 times)
#------------------------------------------

gray = to_gray(False)

torch.manual_seed(opts.seed)

with torch.no_grad():
    for i in range(1):
        for data, data2, path in zip(dataloader_content, dataloader_structure, image_paths):
            name = os.path.basename(path[0])
            id_img, label = data
            id_img_flip = Variable(fliplr(id_img).cuda())
            id_img = Variable(id_img.cuda())
            bg_img, label2 = data2
            if config['single'] == 'gray':
                bg_img = gray(bg_img)
            bg_img = Variable(bg_img.cuda())

            n, c, h, w = id_img.size()
            # Start testing
            c = encode(bg_img)
            f, _ = id_encode(id_img)

            if opts.trainer == 'DGNet':
                outputs = decode(c, f)
                im = recover(outputs[0].data.cpu())
                im = Image.fromarray(im.astype('uint8'))
                ID = name.split('_')
                dst_path = opts.output_folder + '/%03d'%label
                dst_path2 = opts.output_folder2 + '/%03d'%label2
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                if not os.path.isdir(dst_path2):
                    os.mkdir(dst_path2)
                im.save(dst_path + '/%03d_%03d_gan%s.jpg'%(label2, label, name[:-4]))
                im.save(dst_path2 + '/%03d_%03d_gan%s.jpg'%(label2, label, name[:-4]))
            else:
                pass
print('---- start fid evaluation ------')
os.system('cd ../TTUR; python fid.py ../Market/pytorch/train_all ../Market/pytorch/off-gan_id1 --gpu 0')

