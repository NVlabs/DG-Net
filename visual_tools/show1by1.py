"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
import sys
sys.path.append('.')
from utils import get_config
from trainer import DGNet_Trainer, to_gray
import argparse
from torch.autograd import Variable
import sys
import torch
import os
import numpy as np
from torchvision import datasets, transforms
from PIL import Image

name = 'E0.5new_reid0.5_w30000'

if not os.path.isdir('./outputs/%s'%name):
    assert 0, "please change the name to your model name"

parser = argparse.ArgumentParser()
parser.add_argument('--output_folder', type=str, default="./", help="output image path")
parser.add_argument('--input_folder', type=str, default="./visual_data/inputs_many_train", help="input image path")

parser.add_argument('--config', type=str, default='./outputs/%s/config.yaml'%name, help="net configuration")
parser.add_argument('--checkpoint_gen', type=str, default="./outputs/%s/checkpoints/gen_00100000.pt"%name, help="checkpoint of autoencoders")
parser.add_argument('--checkpoint_id', type=str, default="./outputs/%s/checkpoints/id_00100000.pt"%name, help="checkpoint of autoencoders")
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--trainer', type=str, default='DGNet', help="DGNet")


opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)
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
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = datasets.ImageFolder(opts.input_folder, data_transforms)
dataloader_content = torch.utils.data.DataLoader(image_datasets, batch_size=1, shuffle=False, num_workers=1)
dataloader_structure = torch.utils.data.DataLoader(image_datasets, batch_size=16, shuffle=False, num_workers=1)
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

save_path = './visual_data/rainbow'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

im = {}
count = 0
data2 = next(iter(dataloader_structure))
bg_img, _ = data2
gray = to_gray(False)
bg_img = gray(bg_img)
bg_img = Variable(bg_img.cuda())
with torch.no_grad():
    for data in dataloader_content:
        id_img, _ = data
        id_img = Variable(id_img.cuda())
        n, c, h, w = id_img.size()
        # Start testing
        s = encode(bg_img)
        f, _ = id_encode(id_img)
        input1 = recover(data[0].squeeze())
        im[count] = input1
        for i in range(s.size(0)):
            s_tmp = s[i,:,:,:]
            outputs = decode(s_tmp.unsqueeze(0), f)
            tmp = recover(outputs[0].data.cpu())
            pic = Image.fromarray(tmp.astype('uint8'))
            pic.save('%s/rainbow_%d_%d.jpg'%(save_path,i,count))
        count +=1


