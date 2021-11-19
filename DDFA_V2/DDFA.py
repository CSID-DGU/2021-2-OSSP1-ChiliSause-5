import enum
import sys
import argparse
import cv2
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import *
#from utils.render_ctypes import render  # faster
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import *
from utils.pose import *
from utils.serialization import ser_to_ply, ser_to_obj
from utils.serialization import *
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import str2bool


class DDFA:
    def __init__(self):

        cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

        # Init FaceBoxes and TDDFA, recommend using onnx flag
        if True:
            import os
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '4'

            from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
            from TDDFA_ONNX import TDDFA_ONNX

            self.face_boxes = FaceBoxes_ONNX()
            self.tddfa = TDDFA_ONNX(**cfg)
        else:
            gpu_mode = args.mode == 'gpu'
            tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
            face_boxes = FaceBoxes()
    
    #입력: 프레임 이미지
    #출력: roi_box, 외곽선, 정면 바라보는 얼굴 이미지 배열
    def get_faces(self, frame):
        # Detect faces, get 3DMM params and roi boxes
        boxes = self.face_boxes(frame)
        n = len(boxes)
        if n == 0:
            print(f'No face detected, exit')
            sys.exit(-1)
        print(f'Detect {n} faces')

        param_lst, roi_box_lst = self.tddfa(frame, boxes)

        #외곽선 점들 추출
        dense_flag = True
        ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        outlines = draw_landmarks(frame, ver_lst, show_flag=False, dense_flag=dense_flag, wfp=None)

        print(boxes)
        tex = [get_colors(frame), ver_lst[x] for x in range(len(ver_lst))]
        background = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
        render_with_texture_multiple(background, ver_lst, self.tddfa.tri, tex, show_flag=True)
        for v, t in zip(ver_lst, tex):
            render_with_texture_single(background, v, self.tddfa.tri, t, show_flag=True)

        return roi_box_lst, outlines

    #입력: 비식별화 된 이미지 배열, 출력: 원래 각도로 돌려진 이미지 배열
    def restore_faces(self, faces):
        
        pass
