"""
각 프레임에 얼굴을 합성하는 코드
"""
def PlusImage(OriginalFrame, roi_box_lst, restoreImages):
    import cv2
    import numpy as np
    import os.path                 
    from PIL import Image
    
    returnFrame=[]
    roi_box_lst = np.array(roi_box_lst)
    restoreImages = np.array(restoreImages)
    OriginalFrame = np.array(OriginalFrame)
    print(len(OriginalFrame))
    for i in range(0,len(OriginalFrame)) :
        img=OriginalFrame[i]
        for j in range(0,len(restoreImages[i])):
            x=int(roi_box_lst[i][j][0])
            y=int(roi_box_lst[i][j][1])
            width=int(roi_box_lst[i][j][2])-int(roi_box_lst[i][j][0])
            height=int(roi_box_lst[i][j][3])-int(roi_box_lst[i][j][1])
            face=restoreImages[i][j]
            face=cv2.resize(face,dsize=(width,height),interpolation=cv2.INTER_AREA)
            h, w, c = face.shape
            roi = img[y:y+h, x:x+w]
            img1 = face
            img2 = roi
            height1, width1 = img1.shape[:2]
            height2, width2 = img2.shape[:2]
            x = (width2 - width1)//2
            y = height2 - height1
            w = x + width1
            h = y + height1
            chromakey = img1[:10, :10, :]
            offset = 20
            hsv_chroma = cv2.cvtColor(chromakey, cv2.COLOR_BGR2HSV)
            hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            chroma_h = hsv_chroma[:,:,0]
            lower = np.array([chroma_h.min()-offset, 100, 100])
            upper = np.array([chroma_h.max()+offset, 255, 255])
            mask = cv2.inRange(hsv_img, lower, upper)
            mask_inv = cv2.bitwise_not(mask)
            roi = img2[y:h, x:w]
            fg = cv2.bitwise_and(img1, img1, mask=mask_inv)
            bg = cv2.bitwise_and(roi, roi, mask=mask)
            img2[y:h, x:w] = fg + bg
            face=img2         
        returnFrame.append(img)   
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return returnFrame
                
                
                
        

        