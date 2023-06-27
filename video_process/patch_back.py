import os

import numpy as np
from natsort import natsorted
from PIL import Image
from tqdm import tqdm


class TransBack():

    def __init__(self, frame_path, patch_path, crop_list, quad_list, face_list):
        self.get_file(frame_path, patch_path)
        self.crop_list = crop_list
        self.quad_list = quad_list
        self.face_list = face_list

    def get_file(self, frame_path, patch_path):
        filename = natsorted(os.listdir(frame_path))
        frame_list = [os.path.join(frame_path, f) for f in filename]        
        filename = natsorted(os.listdir(patch_path))
        patch_list = [os.path.join(patch_path, f) for f in filename]
        self.frame_list = frame_list
        self.patch_list = patch_list   

    def transback_patch(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        j = 0
        for i, f in enumerate(tqdm(self.frame_list)):
            frame_index = int(f.split('/')[-1].split('.')[0])
            if frame_index not in self.face_list:
                frame = Image.open(f).convert('RGBA')
                frame.save(os.path.join(save_path,f'{str(i)}.png'))
                continue
            
            frame = Image.open(f).convert('RGBA')
            patch = Image.open(self.patch_list[j])
            j += 1
            crop = self.crop_list[i]
            quad = self.quad_list[i]
            quad = list(tuple(x) for x in quad)
            coeffs = self.find_coeffs(quad,
                [(0, 0), (0, 256), (256, 256), (256, 0)]
            )

            x1, y1, x2, y2 = crop[0], crop[1], crop[2], crop[3]

            t_patch = patch.transform((x2-x1, y2-y1), Image.PERSPECTIVE, coeffs, Image.BILINEAR, fillcolor=(255, 255, 255))
            
            t_patch = t_patch.convert('RGBA')
            datas = t_patch.getdata()
            newData = []
            for item in datas:
                if item[0] == 255 and item[1] == 255 and item[2] == 255:
                    newData.append((255, 255, 255, 0))
                else:
                    newData.append(item)
            t_patch.putdata(newData)
            
            frame.paste(t_patch, (x1,y1), t_patch)
            frame.save(os.path.join(save_path,f'{str(i)}.png'))
    
    def find_coeffs(self, pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

        A = np.matrix(matrix, dtype=np.float)
        B = np.array(pb).reshape(8)

        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)