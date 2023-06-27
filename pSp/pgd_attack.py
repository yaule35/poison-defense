import os

import numpy as np
import torch
import torchvision.transforms as transforms
from natsort import natsorted
from PIL import Image
from torch import nn
from tqdm import tqdm

from pSp.utils.common import tensor2im
#%%
#load model
class NewNet(nn.Module):
    def __init__(self, encoder):
        super(NewNet, self).__init__()
        self.encoder = encoder

    def forward(self, x):
        codes = self.encoder(x)
        # w4 : i3 ~ w7 : i6
        out = codes[0][3:7]
        return out
class pSp_encoder():
    def __init__(self, device, encoder):
        self.encoder = encoder
        self.device = device
        
    def pgd_attack(self, sign_map, direction, device, images, eps=8/255, alpha=2/255, iters=4):
        model = self.encoder
        images = images
        ori_image = images.data
        perturb = torch.zeros_like(images).to(device)
        for i in range(iters):
            images.requires_grad = True
            output = model(images)

            model.zero_grad()
            loss = 0
            for i in range(len(sign_map)):
                loss += torch.dot(output[i], sign_map[i]).to(device)
            if direction == -1:
                loss = -loss
            loss.backward()

            adv_img = images + alpha*images.grad.sign()
            eta = torch.clamp(adv_img - ori_image, min=-eps, max=eps)
            perturb += eta
            images = torch.clamp(ori_image + eta, min=-1, max=1).detach_()
        return images   

    def run_attack(self, sign, src_folder, save_folder):
        with open(sign, 'rb') as f:
            mask = np.load(f)
        mask = torch.from_numpy(mask).type(torch.FloatTensor).to(self.device)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        filename = natsorted(os.listdir(src_folder))
        frame_list = [os.path.join(src_folder, f) for f in filename]
        for i, f in enumerate(tqdm(frame_list)):
            image = Image.open(f).convert("RGB")
            transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
            transformed_image = transform(image)
            images = transformed_image.unsqueeze(0).to(self.device).float()
            direction = 1 if i%2 == 0 else -1
            result = self.pgd_attack(mask, direction, self.device, images)
            a = tensor2im(result[0])
            a.save(os.path.join(save_folder,filename[i]))