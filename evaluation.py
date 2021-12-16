import cv2
import numpy as np
import os.path                 
from PIL import Image
import math


from IQA_pytorch import SSIM, DISTS, utils
import torch


def psnr(org, modified):
    mse = np.mean( (org - modified) ** 2 ) #MSE 구하는 코드
    #print("mse : ",mse)

    if mse == 0:
        return 100  
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)) #PSNR구하는 코드

def evaluationFace(OriginalFrame, returnFrame, roi_box_lst):
    cropOrgImges = []
    cropRetImges = []
    for i in range(0,len(OriginalFrame)) :
        img=OriginalFrame[i]
        retimg = returnFrame[i]
        for j in range(0,len(roi_box_lst[i])):
            #좌표와 크기, 얼굴 설정
            x=int(roi_box_lst[i][j][0])
            y=int(roi_box_lst[i][j][1])
            x2=int(roi_box_lst[i][j][2])
            y2=int(roi_box_lst[i][j][3])

            # face=returnFrame[i][j]
            # face=cv2.resize(face,dsize=(width,height),interpolation=cv2.INTER_AREA)
            # h, w, c = face.shape
            
            cropOrgImg=img[y:y2, x:x2]
            cropOrgImges.append(cropOrgImg)

            cropRetImg=retimg[y:y2, x:x2]
            cropRetImges.append(cropRetImg)
            
    evaluation(cropOrgImges, cropRetImges)

def evaluation(OriginalFrame, returnFrame):
    sum=0
    for i in range(0,len(OriginalFrame)):
        #sum+=DISTS1(OriginalFrame[i],returnFrame[i])
        #sum+=SSIM1(OriginalFrame[i],returnFrame[i])
        sum+=psnr(OriginalFrame[i],returnFrame[i])

    average=sum/len(OriginalFrame)
    print("PSNR average : "+str(average))

    sum=0
    for i in range(0,len(OriginalFrame)):
        #sum+=DISTS1(OriginalFrame[i],returnFrame[i])
        sum+=SSIM1(OriginalFrame[i],returnFrame[i])
        #sum+=psnr(OriginalFrame[i],returnFrame[i])
    average=sum/len(OriginalFrame)
    print("SSIM average : "+str(average))

    sum=0
    for i in range(0,len(OriginalFrame)):
        sum+=DISTS1(OriginalFrame[i],returnFrame[i])
        #sum+=SSIM1(OriginalFrame[i],returnFrame[i])
        #sum+=psnr(OriginalFrame[i],returnFrame[i])
    average=sum/len(OriginalFrame)
    print("DISTS average : "+str(average))


def SSIM1(org, modified):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ref = utils.prepare_image(org).to(device)
    # dist = utils.prepare_image(modified).to(device)
    org= Image.fromarray(org)
    org = utils.prepare_image(org.convert('RGB')).to(device)

    modified= Image.fromarray(modified)
    modified = utils.prepare_image(modified.convert('RGB')).to(device)

    model = SSIM()
    score = model(org, modified, as_loss=False)
    return score.item()

def DISTS1(org, modified):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ref = utils.prepare_image(org).to(device)
    # dist = utils.prepare_image(modified).to(device)
    org= Image.fromarray(org)
    org = utils.prepare_image(org.convert('RGB')).to(device)

    modified= Image.fromarray(modified)
    modified = utils.prepare_image(modified.convert('RGB')).to(device)

    model = DISTS().to(device)
    score = model(org, modified, as_loss=False)
    return score.item()