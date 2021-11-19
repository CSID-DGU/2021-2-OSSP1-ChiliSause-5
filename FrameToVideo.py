"""
프레임 단위 영상을 동영상으로 변환하는 코드
resultFps로 초당 프레임을 지정한다 

input data : sample.mp4, frameN.jpg
output data : result.mp4
"""

def FrameToVideo():
    import cv2
    import numpy as np
    import glob
    import os
    import moviepy
    import moviepy.editor

    resultFps=10
    img_array = []
    for filename in glob.glob('./frames/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
 
 
    out = cv2.VideoWriter('./temp.mp4',cv2.VideoWriter_fourcc(*'DIVX'), resultFps, size)
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    videoclip = moviepy.editor.VideoFileClip('./temp.mp4')
    audioclip = moviepy.editor.AudioFileClip('./sample.mp4')
    videoclip.audio = audioclip
    videoclip.write_videofile("./result.mp4")
    os.remove('./temp.mp4')
    print("end")