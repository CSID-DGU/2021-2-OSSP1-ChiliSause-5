"""
Test Code
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def MixImage(OriginalFrame, roi_box_lst, restoreImages):
    import cv2
    import numpy as np
    import os.path                 
    from PIL import Image
    
    returnFrame=[]
    
    img = OriginalFrame[0]
    for index, face in enumerate(restoreImages):
        x=int(roi_box_lst[index][0])
        y=int(roi_box_lst[index][1])
        width=int(roi_box_lst[index][2])-int(roi_box_lst[index][0])
        height=int(roi_box_lst[index][3])-int(roi_box_lst[index][1])

        # height, width = img1.shape[:2]
        # plt.figure(figsize=(12, height / width * 12))
        # plt.imshow(face[..., ::-1])
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # plt.axis('off')
        # plt.show()

        # fig ,ax=plt.subplots()
        # height, width = face.shape[:2]
        # plt.figure(figsize=(width/fig.dpi, height/fig.dpi))
        # plt.imshow(face[..., ::-1])
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # plt.axis('off')
        # plt.show()


        face=cv2.resize(face,dsize=(width,height),interpolation=cv2.INTER_AREA)

        # fig ,ax=plt.subplots()
        # height, width = face.shape[:2]
        # plt.figure(figsize=(width/fig.dpi, height/fig.dpi))
        # plt.imshow(face[..., ::-1])
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # plt.axis('off')
        # plt.show()

        h, w, c = face.shape
        roi = img[y:y+h, x:x+w]
        img1 = face
        img2 = roi
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]
        # print("width:1",width1)
        # print("height1:",height1)
        # print("width:2",width2)
        # print("height2:",height2)
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
    cv2.imshow('append', img)         
    returnFrame.append(img) 



    # for i in OriginalFrame :
    #     img=i
    #     for m, frame_roi in enumerate(roi_box_lst): 
    #         for index, face in enumerate(restoreImages[m]):
    #             x=int(frame_roi[index][0])
    #             y=int(frame_roi[index][1])
    #             width=int(frame_roi[index][2])-int(frame_roi[index][0])
    #             height=int(frame_roi[index][3])-int(frame_roi[index][1])
    #             face=cv2.resize(face,dsize=(width,height),interpolation=cv2.INTER_AREA)
    #             h, w, c = face.shape
    #             roi = img[y:y+h, x:x+w]
    #             img1 = face
    #             img2 = roi
    #             height1, width1 = img1.shape[:2]
    #             height2, width2 = img2.shape[:2]
    #             x = (width2 - width1)//2
    #             y = height2 - height1
    #             w = x + width1
    #             h = y + height1
    #             chromakey = img1[:10, :10, :]
    #             offset = 20
    #             hsv_chroma = cv2.cvtColor(chromakey, cv2.COLOR_BGR2HSV)
    #             hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    #             chroma_h = hsv_chroma[:,:,0]
    #             lower = np.array([chroma_h.min()-offset, 100, 100])
    #             upper = np.array([chroma_h.max()+offset, 255, 255])
    #             mask = cv2.inRange(hsv_img, lower, upper)
    #             mask_inv = cv2.bitwise_not(mask)
    #             roi = img2[y:h, x:w]
    #             fg = cv2.bitwise_and(img1, img1, mask=mask_inv)
    #             bg = cv2.bitwise_and(roi, roi, mask=mask)
    #             img2[y:h, x:w] = fg + bg
    #             face=img2
    #             #img[y:y+h, x:x+w] = face
    #     cv2.imshow('image', img)         
    #     returnFrame.append(img)   
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return returnFrame
                
                
                
        

        