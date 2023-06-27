#%%
import os
import subprocess
import cv2
import dlib
import torch

from pSp.pgd_attack import pSp_encoder
from video_process import patch_back, video_process
video = 'test.mp4'
from torch import nn
class NewNet(nn.Module):
    def __init__(self, encoder):
        super(NewNet, self).__init__()
        self.encoder = encoder

    def forward(self, x):
        codes = self.encoder(x)
        # w4 : i3 ~ w7 : i6
        out = codes[0][3:7]
        return out
#%%
def poison_defense(video):
    #%%
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'Input Video:{video}')

    # frame extraction
    print('1) Frame extracting')
    # work_path = workspace/video_name/
    work_path = 'workspace/' + video.split('.')[0]
    if not os.path.exists(work_path):
        os.makedirs(work_path)
    frame_path =  work_path + '/frame'
    print(f'Frame save path:{frame_path}')
    video_process.video2frame(video, frame_path)
    frames_number = len(os.listdir(frame_path))
    print(f'Extracted frames: {frames_number}')
    print('-' * 50)

    # face extraction and coordinates
    print('2) Face extracting')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_path = work_path + '/face'
    print(f'Face save path:{face_path}')
    crop_list, quad_list, face_list = video_process.frame2face(frame_path, face_path, predictor)
    faces_number = len(os.listdir(face_path))
    print(f'Extracted faces: {faces_number}')
    print('-' * 50)

    # PGD attack
    print('3) Perturbation generating')
    poison_face_path = work_path + '/poison_face'
    print(f'Poison face save path:{poison_face_path}')
    encoder = torch.load('pSp/pretrained_models/encoder.pt')
    encoder.eval()
    encoder.cuda()
    PGD = pSp_encoder(device="cuda", encoder=encoder)
    PGD.run_attack('pSp/latent_mask.npy', face_path, poison_face_path)
    print('Done!')
    print('-' * 50)

    # patch back
    print('4) Patching back')
    transback = patch_back.TransBack(frame_path, poison_face_path, crop_list, quad_list, face_list)
    poison_frame_path = work_path + '/poison_frame'
    print(f'Poison frame save path:{poison_frame_path}')
    transback.transback_patch(poison_frame_path)
    print('-' * 50)

    # video writing
    print('5) Video writing')
    poison_video_name = work_path + '_poison.mp4'
    video_process.frame2video(poison_frame_path, poison_video_name, fps=fps)
    
    # add audio
    result_name = 'protected_' + video
    subprocess.call(['ffmpeg', '-i', f'{poison_video_name}', '-i', f'{video}',
                     '-c:v', 'copy',
                     '-map', '0:v', '-map', '1:a',
                     '-y', f'{result_name}'])
    print(f'Output video:{result_name}')
    print('End!')

    # clean workspace
    workspace = 'workspace/'
    subprocess.call(['rm', '-rf', f'{workspace}'])
    os.mkdir('workspace/')
    return result_name

# %%
