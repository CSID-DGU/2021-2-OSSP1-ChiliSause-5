import enum
import sys
import argparse
import cv2
import yaml
from .utils.rotate_vertices import *
import copy
from .FaceBoxes import FaceBoxes
from .TDDFA import TDDFA
from .utils.render import *
#from utils.render_ctypes import render  # faster
from .utils.depth import depth
from .utils.pncc import pncc
from .utils.uv import *
from .utils.pose import *
from .utils.serialization import ser_to_ply, ser_to_obj
from .utils.serialization import *
from .utils.functions import draw_landmarks, get_suffix
from .utils.tddfa_util import str2bool


class DDFA:
    def __init__(self):

        cfg = yaml.load(open('./DDFA_V2/configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
        self.before_pos_lst = []
        # Init FaceBoxes and TDDFA, recommend using onnx flag
        if False:
            import os
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '4'

            from .FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
            from .TDDFA_ONNX import TDDFA_ONNX

            self.face_boxes = FaceBoxes_ONNX()
            self.tddfa = TDDFA_ONNX(**cfg)
        else:
            gpu_mode = True#args.mode == 'gpu'
            self.tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
            self.face_boxes = FaceBoxes()
    
    #입력: 프레임 이미지
    #출력: roi_box, 외곽선, 정면 바라보는 얼굴 이미지 배열
    def get_faces(self, frame):
        # Detect faces, get 3DMM params and roi boxes
        boxes = self.face_boxes(frame)
        n = len(boxes)
        # if n == 0:
        #     print(f'No face detected, exit')
        #     sys.exit(-1)
        # print(f'Detect {n} faces')

        param_lst, roi_box_lst = self.tddfa(frame, boxes)

        #외곽선 점들 추출
        dense_flag = True
        ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        outlines = draw_landmarks(frame, ver_lst, show_flag=False, dense_flag=dense_flag, wfp=None)

        frontImages = []

        #이미지에서 텍스쳐 추출
        tex = [get_colors(frame, ver_lst[x]) for x in range(len(ver_lst))]

        #배경 생성
        background = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
        for x in background:
            for y in x:
                y[0] = 0
                y[1] = 177
                y[2] = 64
        

        
        
        #포즈 계산
        pose_lst = []
        for param in param_lst:
            P, pose = calc_pose(param)
            pose_lst.append(pose)
            #print(f'yaw: {pose[0]:.1f}, pitch: {pose[1]:.1f}, roll: {pose[2]:.1f}')

        #이전 리스트 기록
        self.before_pos_lst = pose_lst[:]

        for i in range(len(ver_lst)):
            #print(pose_lst[i])
            ver_lst[i] = rotate_v(ver_lst[i], yaw=pose_lst[i][2], pitch=-pose_lst[i][0], roll=pose_lst[i][1])

        #렌더링
        #render_with_texture_multiple(background, ver_lst, self.tddfa.tri, tex, show_flag=True)
        for v, t in zip(ver_lst, tex):
            out = render_with_texture_single(background, v, self.tddfa.tri, t, show_flag=False)
            frontImages.append(out)
        
        #param_lst, roi_box_lst = self.tddfa(out, boxes)
        #for box in roi_box_lst:
        #    frontImages.append(out[int(box[1]):int(box[3]),
        #                         int(box[0]):int(box[2])].copy())

        # print(roi_box_lst)
        roi_box_lst = []
        for i, outline in enumerate(outlines):
            roi_box_lst.append([outline[0].min(), outline[1].min(), outline[0].max(), outline[1].max()] )
        # print(roi_box_lst)

        return roi_box_lst, outlines, frontImages

    #입력: 비식별화 된 이미지 배열, 출력: 원래 각도로 돌려진 이미지 배열
    def restore_faces(self, faces):
        restoreImages = []
        roi_box_lst = []
        outlines = []
        for current_i, face in enumerate(faces):
            frame = face
            
            # Detect faces, get 3DMM params and roi boxes
            boxes = self.face_boxes(frame)
            n = len(boxes)
            # if n == 0:
            #     print(f'No face detected, exit')
            #     sys.exit(-1)
            # print(f'Detect {n} faces')

            param_lst, roi_box_lst = self.tddfa(frame, boxes)

            #외곽선 점들 추출
            dense_flag = True
            ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
            outlines = draw_landmarks(frame, ver_lst, show_flag=False, dense_flag=dense_flag, wfp=None)

            
            #이미지에서 텍스쳐 추출
            tex = [get_colors(frame, ver_lst[x]) for x in range(len(ver_lst))]
            #배경 생성
            background = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
            for x in background:
                for y in x:
                    y[0] = 0
                    y[1] = 177
                    y[2] = 64
            
            for i in range(len(ver_lst)):
                ver_lst[i] = rotate_v(ver_lst[i], yaw=-self.before_pos_lst[current_i][2], pitch=self.before_pos_lst[current_i][0], roll=-self.before_pos_lst[current_i][1])


            #렌더링
            #render_with_texture_multiple(background, ver_lst, self.tddfa.tri, tex, show_flag=True)
            for v, t in zip(ver_lst, tex):
                out = render_with_texture_single(background, v, self.tddfa.tri, t, show_flag=False, isBefore=False)
                restoreImages.append(out)

        #param_lst, roi_box_lst = self.tddfa(out, boxes)
        #for box in roi_box_lst:
        #    frontImages.append(out[int(box[1]):int(box[3]),
        #                         int(box[0]):int(box[2])].copy())
        return roi_box_lst, outlines, restoreImages


if __name__ == "__main__":
    t = euler_translate(0,0,0)

    ddfa = DDFA()
    img = cv2.imread("examples/inputs/images/people.jpg")
    roi, outl, frontImages = ddfa.get_faces(img)
    print(roi)
    pass
    