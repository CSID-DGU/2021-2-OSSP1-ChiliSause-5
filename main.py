import DDFA_V2
from DDFA_V2.DDFA import DDFA
from pixel2style2pixel.scripts.style_mixing import StyleMix

import cv2
import VideoToFrame
import PlusImage
import FrameToVideo



#ddfa = DDFA()
StyleGan = StyleMix()

if __name__ == "__main__":
    #pipeline1 : sample.mp4파일을 입력으로 받아, OriginalFrame에 프레임을 저장한다.
    #OriginalFrame=VideoToFrame.VideoToFrame()

    """
    for img in OriginalFrame: 
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """


    #for currentFrame in OriginalFrame:
        # pipeline2
        #roi_box_lst, outlines, frontImages = ddfa.get_faces(frame=currentFrame)
        
        #piplen3: deidentification by StylrGan
        # StyleGan.set_faceImgInput(frontImages)      #stylegan input setting
        # StyleGan.mix()                              #run stylegan
        # deIdentificationImgArr = StyleGan.get_face()#stylegan output
        
        #styleGan .pt file
        #https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view

    
    
    """
    #pipeline5 : 비식별화된 얼굴 이미지를 원본 프레임에 합성
    roi_box_lst, outlines, restoreImages=restore_faces(self, faces)
    Frame=PlusImage(roi_box_lst, restoreImages)
    """

    """
    #pipeline7 : 후처리된 프레임과 원본동영상으로 결과 동영상 생성
    FrameToVideo(Frame)
    """