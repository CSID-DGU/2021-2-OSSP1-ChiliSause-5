import numpy as np

def euler_rotate(yaw, pitch, roll):
    #yaw pitch roll : 각도로 입력

    #라디안 변환
    yaw *= np.pi/180
    pitch *= np.pi/180
    roll *= np.pi/180

    Rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [          0,            0, 1]])
    Ry_pitch = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx_roll = np.array([
        [1,            0,             0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]])
    # R = RzRyRx
    rotMat = np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll))
    return rotMat

def euler_translate(x, y, z):
    pass

def rotate(vertics, yaw, pitch, roll):
    #vertex와 회전값 입력
    rotated_vertics = np.dot(euler_rotate(yaw, pitch, roll),vertics)
    return rotated_vertics