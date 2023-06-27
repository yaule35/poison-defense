#%%
import os
import pprint
from argparse import Namespace

import natsort
import numpy as np
import torch
import torchvision.transforms as transforms
from natsort import natsorted
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pSp.models.psp import pSp
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
encoder = torch.load('pretrained_models/model')
encoder.eval()
encoder.cuda()
device="cuda"

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image).float()
        return tensor_image
trsfm = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
my_dataset = CustomDataSet('00000', transform=trsfm)
train_loader = DataLoader(my_dataset , batch_size=5, shuffle=False, num_workers=2, drop_last=True)
def uap_sgd(model, loader, sign_map, nb_epoch, eps, direction=1, step_decay = 0.8, uap_init = None, layer_name=None):
    '''
    INPUT
    model       model
    loader      dataloader
    nb_epoch    number of optimization epochs
    eps         maximum perturbation value (L-infinity) norm
    direction   sign direction for maximum of minimum loss
    loss_fn     custom loss function (default is CrossEntropyLoss)
    uap_init    custom perturbation to start from (default is random vector with pixel values {-eps, eps})
    
    OUTPUT
    delta.data  adversarial perturbation
    losses      losses per iteration
    '''
    _, x_val = next(enumerate(loader))
    batch_size = len(x_val)
    if uap_init is None:
        batch_delta = torch.zeros_like(x_val) # initialize as zero vector
    else:
        batch_delta = uap_init.unsqueeze(0).repeat([batch_size, 1, 1, 1])
    delta = batch_delta[0]
    losses = []
    with open(sign_map, 'rb') as f:
        sign = np.load(f)
    sign = torch.from_numpy(sign).type(torch.FloatTensor).to(device)
    # loss function
    if layer_name is None:
        def clamped_loss(output, sign):
            loss = 0
            for i in range(len(sign)):
                loss += torch.dot(output[i], sign[i]).to(device)
            return loss
       
    ## layer maximization attack
    # else:
    #     def get_norm(self, forward_input, forward_output):
    #         global main_value
    #         main_value = torch.norm(forward_output, p = 'fro')
    #     for name, layer in model.named_modules():
    #         if name == layer_name:
    #             handle = layer.register_forward_hook(get_norm)

    batch_delta.requires_grad = True
    for epoch in range(nb_epoch):
        print('epoch %i/%i' % (epoch + 1, nb_epoch))
        
        # perturbation step size with decay
        eps_step = eps * step_decay
        
        for i, x_val in enumerate(tqdm(loader, leave=False)):
            batch_delta.data = delta.unsqueeze(0).repeat([x_val.shape[0], 1, 1, 1])
            
            perturbed = torch.clamp((x_val + batch_delta).to(device), -1, 1)
            outputs = model(perturbed)
            model.zero_grad()
            # loss function value
            loss = clamped_loss(outputs, sign)
            if direction == -1:
                loss = -loss
            losses.append(torch.mean(loss))
            loss.backward()
            # batch update
            grad_sign = batch_delta.grad.data.mean(dim = 0).sign()
            delta = delta + grad_sign * eps_step
            delta = torch.clamp(delta, -eps, eps)
            batch_delta.grad.data.zero_()
    
    return delta.data, losses
#%%
nb_epoch = 10
eps = 8 / 255
step_decay = 0.7

#%%
direction = 1
sign_map = '0515_latent_selection_w4.npy'
uap_plus, losses_plus = uap_sgd(encoder, train_loader, sign_map, nb_epoch, eps, direction, step_decay)
#transpose (3, 256, 256) tp (256, 256, 3)
# t255 = uap_plus.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()*255
t = uap_plus.cpu().detach().numpy()
# np.save('0605_uap_plus_1024_255.npy',t255)
np.save('C_uap_plus_w4.npy',t)
#%%
direction = -1
sign_map = '0515_latent_selection_w4.npy'
uap_minus, losses_minus = uap_sgd(encoder, train_loader, sign_map, nb_epoch, eps, direction, step_decay)
#transpose (3, 256, 256) tp (256, 256, 3)
# t255 = uap_minus.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()*255
t = uap_minus.cpu().detach().numpy()
# np.save('0605_uap_minus_1024_255.npy',t255)
np.save('C_uap_minus_w4.npy',t)
#%%
del uap_minus, losses_minus, t
#%%
direction = -1
uap_minus, losses_minus = uap_sgd(encoder, train_loader, nb_epoch, eps, direction, step_decay)

t255 = uap_minus.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()*255
t = uap_minus.cpu().detach().numpy()
np.save('elon_uap_w4_minus_255.npy',t255)
np.save('elon_uap_w4_minus.npy',t)
#%%
a = tensor2im(uap_plus)
a.save("uap_plus.jpg")
#%%
with open(r'losses_plus.txt', 'w') as fp:
    for item in losses_plus:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')
# %% save perturbations

#%%
with open('uap_plus_w4.npy', 'rb') as f:
    lo = np.load(f)
lo/255
# %% Paste Universal patch

def uap_patch(plus_patch, minus_patch, src_folder, save_folder):
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
        transformed_image = transform(image).float()
        if i%2 == 0:
            images = np.clip(transformed_image+plus_patch, -1, 1)
        else:
            images = np.clip(transformed_image+minus_patch, -1, 1)
        a = tensor2im(images)
        a.save(os.path.join(save_folder,filename[i]))

with open('C_uap_plus_w4.npy', 'rb') as f:
    p = np.load(f)
plus_patch = torch.from_numpy(p)
with open('C_uap_minus_w4.npy', 'rb') as f:
    m = np.load(f)
minus_patch = torch.from_numpy(m)
uap_patch(plus_patch, minus_patch, 'data_dst_0326_face', '0610_w47uap_w4c')
#%%
a = Image.open('00001.png')
transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
transformed_image = transform(a).float()
transformed_image.shape
# %%
with open('uap_plus_w4.npy', 'rb') as f:
    p = np.load(f)
plus_patch = torch.from_numpy(p/255)
plus_patch.shape
# %%
