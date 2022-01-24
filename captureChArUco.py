import cv2 as cv
import numpy as np
from cv2 import aruco
import CameraModule.CameraModule as CM
import os

IMGSIZE = (640,480)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

def main():
    if not os.path.isdir('./Aruco_captures'):
        os.mkdir('./Aruco_captures')
    if not os.path.isdir('./Aruco_datas'):
        os.mkdir('./Aruco_datas')
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)
    imboard = board.draw((2000, 2000))
    if not os.path.exists('./chessboard.tiff'):
        print('Chessboard has created.')
        cv.imwrite("./chessboard.tiff", imboard)
    stereoCam = CM.StereoCamera(IMGSIZE)
    allCorners = []
    allIds = []
    frame = stereoCam.GetFrame()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(gray, aruco_dict)
    if(len(corners) > 0):
        for corner in corners:
            cv.cornerSubPix(gray, corner, (3,3), (-1,-1),criteria)
            ret, charucoCorners, charucoIds = cv.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if charucoCorners is not None and charucoIds is not None and len(charucoCorners) > 3:
                allCorners.append(charucoCorners)
                allIds.append(charucoIds)
    # TODO : 검출된 charuco와 Id값 저장하기

if __name__ == "__main__":
    main()