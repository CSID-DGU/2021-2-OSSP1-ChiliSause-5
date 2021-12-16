import numpy as np

#yaw pitch roll : 각도로 입력
def euler_rotate(yaw, pitch, roll):


    #라디안 변환
    yaw *= np.pi/180
    pitch *= np.pi/180
    roll *= np.pi/180

    Rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, 0],
        [np.sin(yaw),  np.cos(yaw), 0, 0],
        [          0,            0, 1, 0],
        [          0,            0, 0, 1]], dtype=np.float32)
    Ry_pitch = np.array([
        [ np.cos(pitch), 0,-np.sin(pitch), 0],
        [             0, 1,             0, 0],
        [ np.sin(pitch), 0, np.cos(pitch), 0],
        [          0,    0,             0, 1]], dtype=np.float32)
    Rx_roll = np.array([
        [1,            0,             0, 0],
        [0, np.cos(roll), -np.sin(roll), 0],
        [0, np.sin(roll),  np.cos(roll), 0],
        [0,            0,             0, 1]], dtype=np.float32)
    # R = RzRyRx
    rotMat = np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll))
    return rotMat

def euler_translate(x, y, z):
    move = np.array([
        [1,0,0,x],
        [0,1,0,y],
        [0,0,1,z],
        [0,0,0,1]
    ], dtype=np.float32)
    return move

def meanofp(axis):
    return (axis.min()+axis.max())/2

#vertex와 회전값 입력
#vertex 입력 형식
#x 0 ...
#y 0 ...
#z 0 ...
def rotate_v(vertics, yaw, pitch, roll):
    vertics_ = np.vstack([vertics, np.array( 
                        [1 for x in range(vertics.shape[1])], 
                        dtype=np.float32 )])
    #중심점으로 이동후 변환
    mvx = meanofp(vertics[0])
    mvy = meanofp(vertics[1])
    mvz = meanofp(vertics[2])
    rotate_point = np.dot(euler_translate(mvx, mvy, mvz), 
                    np.dot(euler_rotate(yaw, pitch, roll), euler_translate(-mvx, -mvy, -mvz)))
    rotated_vertics = np.dot(rotate_point, vertics_)

    return np.delete(rotated_vertics, (3), axis=0)