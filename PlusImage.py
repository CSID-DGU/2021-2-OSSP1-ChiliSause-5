"""
각 프레임에 얼굴을 합성하는 코드

input data : 100000+N.jpg, 100000+N_faceM.jpg, face_coordinate.txt
output data : 100000+N.jpg
"""
def PlusImage():
    import cv2
    import numpy as np
    import os.path                 
    from PIL import Image
    i=0

    f = open("./face_coordinate.txt", 'r')
    while 1:
        fileNum=100000+i
        fileName = "./frames/"+str(fileNum)+".jpg"
        if not os.path.isfile(fileName):
            break
        img = cv2.imread(fileName, 1)
    
        j=0
        while 1:
            faceName="./faces/"+str(fileNum)+"_face"+str(j)+".jpg"
            if not os.path.isfile(faceName):
                break
            x=int(f.readline())
            y=int(f.readline())
            width=int(f.readline())
            height=int(f.readline())
            print(faceName,x,y,width,height)
            face = cv2.imread(faceName, 1)
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
            #img[y:y+h, x:x+w] = face
            j+=1    
        cv2.imwrite(fileName, img)   
        i+=1
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
