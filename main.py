import DDFA_V2
from DDFA_V2.DDFA import DDFA
from pixel2style2pixel.scripts.style_mixing import StyleMix
import matplotlib.pyplot as plt
import cv2
import VideoToFrame
import PlusImage
import FrameToVideo
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


ddfa = DDFA()
StyleGan = StyleMix()

if __name__ == "__main__":
    #pipeline1 : sample.mp4파일을 입력으로 받아, OriginalFrame에 프레임을 저장한다.
    OriginalFrame=VideoToFrame.VideoToFrame()

    """
    for img in OriginalFrame: 
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """
    roi_box_lst, outlines, frontImages = ddfa.get_faces(frame=OriginalFrame[0])
    # img=frontImages[0]
    # height, width = img.shape[:2]
    # plt.figure(figsize=(12, height / width * 12))

    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # plt.axis('off')

    # plt.imshow(img[..., ::-1])
    # plt.show()
    # print(frontImages[0].shape)
    # input()

#     #piplen3: deidentification by StyleGan
    StyleGan.set_faceImgInput(frontImages)      #stylegan input setting
    StyleGan.mix()                              #run stylegan
    deIdentificationImgArr = StyleGan.get_face()#stylegan output
    
    # img=deIdentificationImgArr[0]
    # height, width = img.shape[:2]
    # plt.figure(figsize=(12, height / width * 12))

    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # plt.axis('off')

    # plt.imshow(img[..., ::-1])
    # plt.show()
    # print(img.shape)
    # input()
    
    #pipeline4
    ddfa.restore_faces(deIdentificationImgArr)

    # for currentFrame in OriginalFrame:
    #     #pipeline2
    #     roi_box_lst, outlines, frontImages = ddfa.get_faces(frame=currentFrame)
    #     plt.imshow(frontImages[0])
    #     cv2.imshow('test',frontImages[0])
    #     print(frontImages[0])

    #     #piplen3: deidentification by StyleGan
    #     # StyleGan.set_faceImgInput(frontImages)      #stylegan input setting
    #     # StyleGan.mix()                              #run stylegan
    #     # deIdentificationImgArr = StyleGan.get_face()#stylegan output
    #     # StyleGan.show_face()

    
    
    """
    #pipeline5 : 비식별화된 얼굴 이미지를 원본 프레임에 합성
    roi_box_lst, outlines, restoreImages=restore_faces(self, faces)
    Frame=PlusImage(roi_box_lst, restoreImages)
    """

    """
    #pipeline7 : 후처리된 프레임과 원본동영상으로 결과 동영상 생성
    FrameToVideo(Frame)
    """