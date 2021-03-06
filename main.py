from tqdm.std import tqdm
import DDFA_V2
from DDFA_V2.DDFA import DDFA
from pixel2style2pixel.scripts.style_mixing import StyleMix
import imageMix
import matplotlib.pyplot as plt
import cv2
import VideoToFrame
import PlusImage
import Plusimage_old
import evaluation
import FrameToVideo
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time


start = time.time()

ddfa = DDFA()
StyleGan = StyleMix()

def outlineToroi(outline):
    roi = []
    for out in outline:
        out = np.transpose(out)
        roi.append([out[0].min()/10, out[1].min()/10, out[0].max()/10, out[1].max()/10])
    return roi


if __name__ == "__main__":

    #pipeline1 : sample.mp4파일을 입력으로 받아, OriginalFrame에 프레임을 저장한다.
    OriginalFrame=VideoToFrame.VideoToFrame()

    """
    for img in OriginalFrame: 
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """
    # roi_box_lst, outlines, frontImages = ddfa.get_faces(frame=OriginalFrame[0])
    # # print(outlines)
    # # roi_box_lst = outlineToroi(outlines)
    # # print(roi_box_lst)

    # StyleGan.set_faceImgInput(frontImages)
    # StyleGan.mix() 
    # deIdentificationImgArr = StyleGan.get_face()

    # roi_box_lst_, outlines, restoreImages=ddfa.restore_faces(deIdentificationImgArr)

    # returnFrame=imageMix.MixImage(OriginalFrame,roi_box_lst,restoreImages)

    roi_box_lst_arr = []
    restoreImages_arr = []
    for currentFrame in tqdm(OriginalFrame):
        #pipeline2
        # height, width = currentFrame.shape[:2]
        # plt.figure(figsize=(12, height / width * 12))
        # plt.imshow(currentFrame[..., ::-1])
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # plt.axis('off')
        # plt.show()
        roi_box_lst, outlines, frontImages = ddfa.get_faces(frame=currentFrame)
        # height, width = frontImages[0].shape[:2]
        # plt.figure(figsize=(12, height / width * 12))
        # plt.imshow(frontImages[0][..., ::-1])
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # plt.axis('off')
        # plt.show()
        roi_box_lst = outlineToroi(outlines)

        print(len(frontImages))
        #piplen3: deidentification by StyleGan
        StyleGan.set_faceImgInput(frontImages)      #stylegan input setting
        StyleGan.mix()                              #run stylegan
        deIdentificationImgArr = StyleGan.get_face()#stylegan output
        #print(len(deIdentificationImgArr))
        # for img in deIdentificationImgArr:
        #     height, width = img.shape[:2]
        #     plt.figure(figsize=(12, height / width * 12))
        #     plt.imshow(img[..., ::-1])
        #     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #     plt.axis('off')
        #     plt.show()
        #pipeline4
        roi_box_lst_, outlines, restoreImages=ddfa.restore_faces(deIdentificationImgArr)
        
        # for res in restoreImages:
        #     print(res.shape)
        #     height, width = res.shape[:2]
        #     plt.figure(figsize=(12, height / width * 12))
        #     plt.imshow(res[..., ::-1])
        #     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #     plt.axis('off')
        #     plt.show()

        #     # height, width = img.shape[:2]
        #     #     plt.figure(figsize=(12, height / width * 12))

        #     #     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #     #     plt.axis('off')

        #     #     plt.imshow(img[..., ::-1])
        #     #     plt.show()

        # input()
        
        roi_box_lst_arr.append(roi_box_lst)
        restoreImages_arr.append(restoreImages)


    
    
#     [[554.0776062011719, 102.47742233276367, 665.0776062011719, 213.47742233276367], [377.00584411621094, 135.89453979492188, 517.0058441162109, 275.8945397949219], [166.24088287353516, 153.28521377563476, 297.24088287353516, 284.28521377563476]]
# [[577.6594, 112.033875, 647.5167, 188.67853], [405.15213, 155.23764, 484.96057, 245.54279], [187.72116, 160.8189, 269.29086, 250.4819]]
    
    #pipeline5 : 비식별화된 얼굴 이미지를 원본 프레임에 합성하여 returnFrame에 저장한다.
    
    """
    for i in restoreImages_arr: 
        for j in i:
            cv2.imshow('image',j)
            cv2.waitKey(2000)
    cv2.destroyAllWindows()
    """
    returnFrame_=PlusImage.PlusImage1(OriginalFrame, roi_box_lst_arr, restoreImages_arr)
    #returnFrame_=Plusimage_old.PlusImage(OriginalFrame, roi_box_lst_arr, restoreImages_arr)
    
    """
    print("returnFrame length : ")
    print(len(returnFrame_))
    

    
    for i in range(0,len(returnFrame_)): 
        cv2.imshow('image', returnFrame_[i])
        cv2.waitKey(2000)
    cv2.destroyAllWindows()
    """
    
    
    
   
    
    
    

    
    #pipeline7 : 후처리된 프레임(returnFrame_)과 원본동영상으로 결과 동영상(result.mp4)을 생성한다.
    FrameToVideo.FrameToVideo(returnFrame_)
    

    # 성능평가 후 결과 출력
    #psnr=evaluation.evaluation(OriginalFrame,returnFrame_)
    evaluation.evaluationFace(OriginalFrame, returnFrame_, roi_box_lst_arr)
    print("실행시간 :", time.time() - start)