"""
동영상을 프레임 단위로 추출하는 코드
resultFps로 초당 프레임을 지정한다

input data : sample.mp4 
output data : 100000+N.jpg
"""
def VideoToFrame():
    import cv2

    resultFps=0.1
    returnImage=[]
    vidcap = cv2.VideoCapture('./sample.mp4')

    totalFrame = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    print('total frame:', totalFrame)
    print('FPS:', fps)

    saveCount = 0
    frameCount=0
    for i in range(0, int(totalFrame)):   
        ret, image = vidcap.read()
        if(int(vidcap.get(1))==1+int(round(saveCount*fps/resultFps))):
            #cv2.imwrite("./frames/%d.jpg" % (100000+saveCount), image)
            returnImage.append(image)
            #print('Saved frame%d.jpg' % saveCount)
            saveCount += 1
        

    print(" VideoToFrame end")        
    vidcap.release()
    return returnImage
