import DDFA_V2
from DDFA_V2.DDFA import DDFA
from pixel2style2pixel.scripts.style_mixing import StyleMix
import imageMix
import matplotlib.pyplot as plt
import cv2
import VideoToFrame
import PlusImage
import FrameToVideo
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

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
    for currentFrame in OriginalFrame:
        #pipeline2
        roi_box_lst, outlines, frontImages = ddfa.get_faces(frame=currentFrame)
        # plt.imshow(frontImages[0])
        # cv2.imshow('test',frontImages[0])
        # print(frontImages[0])
        #roi_box_lst = outlineToroi(outlines)

        #piplen3: deidentification by StyleGan
        StyleGan.set_faceImgInput(frontImages)      #stylegan input setting
        StyleGan.mix()                              #run stylegan
        deIdentificationImgArr = StyleGan.get_face()#stylegan output

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


    
    
    
    #pipeline5 : 비식별화된 얼굴 이미지를 원본 프레임에 합성 
    returnFrame_=PlusImage.PlusImage(OriginalFrame, roi_box_lst_arr, restoreImages_arr)
    print("returnFrame length : ")
    print(len(returnFrame_))
    for i in range(0,len(returnFrame_)):
        cv2.imshow('image', returnFrame_[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    
    
    

    """
    #pipeline7 : 후처리된 프레임과 원본동영상으로 결과 동영상 생성
    FrameToVideo(Frame)
    """