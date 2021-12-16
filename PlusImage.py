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
            #좌표와 크기, 얼굴 설정
            x=int(roi_box_lst[i][j][0])
            y=int(roi_box_lst[i][j][1])
            width=int(roi_box_lst[i][j][2])-int(roi_box_lst[i][j][0])
            height=int(roi_box_lst[i][j][3])-int(roi_box_lst[i][j][1])
            face=restoreImages[i][j]
            face=cv2.resize(face,dsize=(width,height),interpolation=cv2.INTER_AREA)
            h, w, c = face.shape
            #원본 이미지에서 roi를 잘라냄
            roi = img[y:y+h, x:x+w]
            img1 = face
            img2 = roi
            height1, width1 = img1.shape[:2]
            height2, width2 = img2.shape[:2]
            x = (width2 - width1)//2
            y = height2 - height1
            w = x + width1
            h = y + height1
            #roi와 face를 합침
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
        #img를 return에 추가함
        returnFrame.append(img)   
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return returnFrame
                
                
                
        
def PlusImage1(OriginalFrame, roi_box_lst, restoreImages):
    import cv2
    import numpy as np
    import os.path                 
    from PIL import Image
    from preprocessing import PreProcessing
    from kernel import Poisson
    from tqdm.std import tqdm
    
    pre = PreProcessing()
            
    returnFrame=[]
    roi_box_lst = np.array(roi_box_lst)
    restoreImages = np.array(restoreImages)
    OriginalFrame = np.array(OriginalFrame)
    print(len(OriginalFrame))
    for i, img in enumerate(tqdm(OriginalFrame)):
        for j, face in enumerate(restoreImages[i]):
            # cv2.imshow('img', img)
            # cv2.waitKey(10)
            # cv2.destroyAllWindows()
            x=int(roi_box_lst[i][j][0])
            y=int(roi_box_lst[i][j][1])
            width=int(roi_box_lst[i][j][2])-int(roi_box_lst[i][j][0])
            height=int(roi_box_lst[i][j][3])-int(roi_box_lst[i][j][1])
            centerX=x+width/2
            centerY=y+height/2
            face=cv2.resize(face,dsize=(width,height),interpolation=cv2.INTER_AREA)       
            # cv2.imshow('face', face)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            chromakey = face[:10, :10, :]
            offset = 20
            hsv_chroma = cv2.cvtColor(chromakey, cv2.COLOR_BGR2HSV)
            hsv_img = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            chroma_h = hsv_chroma[:,:,0]
            lower = np.array([chroma_h.min()-offset, 100, 100])
            upper = np.array([chroma_h.max()+offset, 255, 255])
            mask = cv2.inRange(hsv_img, lower, upper)
            mask_inv = cv2.bitwise_not(mask)
            mask = mask_inv                              #마스크(grascale필수, 흑,백 => 흑/안보이게, 백/부분만 추출),src의 마스크
            
            fx=width-12
            fy=height-12
            mask = cv2.resize(mask, dsize=(fx,fy),interpolation=cv2.INTER_NEAREST)
            
            mask = cv2.copyMakeBorder(mask, 6,6,6,6, cv2.BORDER_CONSTANT, value=0)
            cv2.imwrite("./maskTest/test.jpg",mask)
            # selectedImg = cv2.copyTo(face, mask)
            # cv2.imshow('select_mask', selectedImg)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            xy = (centerY,centerX)
            # xy = (120,100)                                                                                              #dst의 복사 중심좌표 (y,x)
            pre.selectMask(face, img, mask)                                                                                #마스크에 대한 처리 진행
            #pre.select(src, dst)

            #retImg = Poisson.seamlessClone(src, dst, pre.selectedMask, pre.selectedPoint, Poisson.MIXED_CLONE)
            img = Poisson.seamlessClone(face,img,mask,xy,Poisson.NORMAL_CLONE)
            # cv2.imshow('result', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            
            
            #좌표와 크기, 얼굴 설정
            # x=int(roi_box_lst[i][j][0])
            # y=int(roi_box_lst[i][j][1])
            # width=int(roi_box_lst[i][j][2])-int(roi_box_lst[i][j][0])
            # height=int(roi_box_lst[i][j][3])-int(roi_box_lst[i][j][1])
            # face=restoreImages[i][j]
            # face=cv2.resize(face,dsize=(width,height),interpolation=cv2.INTER_AREA)
            # h, w, c = face.shape
            # centerX=x+width/2
            # centerY=y+height/2
            
            # #원본 이미지에서 roi를 잘라냄
            # roi = img[y:y+h, x:x+w]
            # img1 = face
            # img2 = roi
            # height1, width1 = img1.shape[:2]
            # height2, width2 = img2.shape[:2]
            # x = (width2 - width1)//2
            # y = height2 - height1
            # w = x + width1
            # h = y + height1
            # #roi와 face를 합침
            # chromakey = img1[:10, :10, :]
            # offset = 20
            # hsv_chroma = cv2.cvtColor(chromakey, cv2.COLOR_BGR2HSV)
            # hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            # chroma_h = hsv_chroma[:,:,0]
            # lower = np.array([chroma_h.min()-offset, 100, 100])
            # upper = np.array([chroma_h.max()+offset, 255, 255])
            # mask = cv2.inRange(hsv_img, lower, upper)
            # mask_inv = cv2.bitwise_not(mask)
            # roi = img2[y:h, x:w]
            # fg = cv2.bitwise_and(img1, img1, mask=mask_inv)
            # bg = cv2.bitwise_and(roi, roi, mask=mask)
            # img2[y:h, x:w] = fg + bg
            # face=img2         
        #img를 return에 추가함
        returnFrame.append(img)
    
    return returnFrame
        