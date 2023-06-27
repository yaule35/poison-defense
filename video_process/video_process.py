import os

import cv2
from natsort import natsorted
from tqdm import tqdm

from video_process.align_all_parallel import align_face

# Extract frames from video
def video2frame(video_name, save_pth):
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    v = cv2.VideoCapture(video_name)
    fps = v.get(cv2.CAP_PROP_FPS)
    success,image = v.read()
    count = 0
    while success:
        cv2.imwrite("%s/%d.png" % (save_pth, count), image)     # save frame as JPEG file      
        success,image = v.read()
        count += 1
    print('Frames extracted!')

# %% Align face from frame, and record crop and quad transform data
def frame2face(src_pth, save_pth, predictor):
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    filename = natsorted(os.listdir(src_pth))
    frame_list = [os.path.join(src_pth, f) for f in filename]
    quad_list = []
    crop_list = []
    face_list = []
    for i, f in enumerate(tqdm(frame_list)):
        face_result = align_face(filepath=f, predictor=predictor)
        if face_result:
            face_list.append(i)
            aligned_image, crop, quad = face_result[0], face_result[1], face_result[2]
            aligned_image.save(os.path.join(save_pth,filename[i]))
            quad = (quad + 0.5)
            crop_list.append(list(crop))
            quad_list.append(quad)
        else:
            crop_list.append(None)
            quad_list.append(None)
    print('Faces extracted!')
    return crop_list, quad_list, face_list

# image to video
def frame2video(image_folder, video_name, fps):
    images = [img for img in natsorted(os.listdir(image_folder)) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()
    print('Video writing done!')