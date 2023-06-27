import os
import subprocess
import torch
import cv2
import dlib

from natsort import natsorted
from tqdm import tqdm

from pSp.pgd_attack import pSp_encoder
from video_process import patch_back, video_process
from video_process.align_all_parallel import align_face
def poison_defense(video):
    torch.multiprocessing.set_start_method('spawn')
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    work_path = 'workspace/' + video.split('.')[0]
    if not os.path.exists(work_path):
        os.makedirs(work_path)
    frame_path =  work_path + '/frame'
    print(f'Frame save path:{frame_path}')
    video_process.video2frame(video, frame_path)
    frames_number = len(os.listdir(frame_path))
    print(f'Extracted frames: {frames_number}')
    print('-' * 25)

    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    filename = natsorted(os.listdir(frame_path))
    frame_list = [os.path.join(frame_path, f) for f in filename]
    quad_list = []
    crop_list = []
    for i, f in enumerate(frame_list):
        face_result = align_face(filepath=f, predictor=predictor)
        frame_list = [os.path.join(src_pth, f) for f in filename]
        face_result = align_face(filepath=f, predictor=predictor)
        if face_result:
            aligned_image, crop, quad = face_result[0], face_result[1], face_result[2]
            quad = (quad + 0.5)
            crop_list.append(list(crop))
            quad_list.append(quad)
        else:
            crop_list.append(None)
            quad_list.append(None)