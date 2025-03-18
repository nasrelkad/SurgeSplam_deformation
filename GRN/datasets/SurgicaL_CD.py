from __future__ import absolute_import, division, print_function

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" # allow cv2 to read EXR images
import numpy as np
import PIL.Image as pil
import cv2
import json

from .mono_dataset import MonoDataset


class SyntheticSurgicalDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(SyntheticSurgicalDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.82, 0, 0.5, 0],
                           [0, 1.02, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # self.full_res_shape = (1280, 1024)
        # self.side_map = {"l": "left", "r": "right"}
        self.style = 'style_00'
    def check_depth(self):
        
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))
        
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class SyntheticSurgicalDepthDataset(SyntheticSurgicalDataset):
    def __init__(self, *args, **kwargs):
        super(SyntheticSurgicalDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index):
        f_str = "img{:05d}{}".format(frame_index, self.img_ext)
        
        
        image_path = os.path.join(
            self.data_path, 'stylernd', folder,self.style, f_str) 

        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "depth{:05d}.exr".format(frame_index)

        
        depth_path = os.path.join(
            self.data_path, 'depths/simulated',folder, "depths", f_str)

        depth_gt = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
    



