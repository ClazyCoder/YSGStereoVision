import cv2 as cv
import numpy as np
from cv2 import aruco
import CameraModule.CameraModule as CM
import os

IMGSIZE = (640,480)

def main():
    if not os.path.isdir('./Aruco_captures'):
        os.mkdir('./Aruco_captures')
    if not os.path.isdir('./Aruco_datas'):
        os.mkdir('./Aruco_datas')
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)
    imboard = board.draw((2000, 2000))
    cv.imwrite("./chessboard.tiff", imboard)
    stereoCam = CM.StereoCamera(IMGSIZE)
    allCorners = []
    allIds = []
    decimator = 0
    frame = stereoCam.GetFrame()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(gray, aruco_dict)
    if(len(corners) > 0):
        # TODO : Save Points.
        pass

if __name__ == "__main__":
    main()