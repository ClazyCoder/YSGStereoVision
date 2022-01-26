import cv2 as cv
import numpy as np
from cv2 import aruco
import CameraModule.CameraModule as CM
import os

IMGSIZE = (640,480)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

def main():
    if not os.path.isdir('./ChAruco_captures'):
        os.mkdir('./ChAruco_captures')
    if not os.path.isdir('./ChAruco_datas'):
        os.mkdir('./ChAruco_datas')
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)
    imboard = board.draw((2000, 2000))
    if not os.path.exists('./chessboard.tiff'):
        print('Chessboard has created.')
        cv.imwrite("./chessboard.tiff", imboard)
    stereoCam = CM.StereoCamera(IMGSIZE)
    allCornersLeft = []
    allCornersRight = []
    allIds = []
    objpoints= []
    isCalibrating = False
    while True:
        leftFrame, rightFrame = stereoCam.GetFrame()
        grayLeft = cv.cvtColor(leftFrame, cv.COLOR_BGR2GRAY)
        grayRight = cv.cvtColor(rightFrame, cv.COLOR_BGR2GRAY)
        if isCalibrating:
            cornersLeft, idsLeft, rejectedImgPointsLeft = cv.aruco.detectMarkers(grayLeft, aruco_dict)
            cornersRight, idsRight, rejectedImgPointsRight = cv.aruco.detectMarkers(grayRight, aruco_dict)
            if(len(cornersLeft) > 0 and len(cornersRight) > 0):
                for corner in cornersLeft:
                    cv.cornerSubPix(grayLeft, corner, (3,3), (-1,-1),criteria)
                    ret, charucoCorners, charucoIds = cv.aruco.interpolateCornersCharuco(cornersLeft,idsLeft,grayLeft,board)
                    if ret:
                        allCornersLeft.append(charucoCorners)
                        allIds.append(charucoIds)
                for corner in cornersRight:
                    cv.cornerSubPix(grayRight,corner, (3,3),(-1,-1),criteria)
                    ret, charucoCorners, charucoIds = cv.aruco.interpolateCornersCharuco(cornersRight,idsRight,grayRight,board)
                    if ret:
                        allCornersRight.append(charucoCorners)
                newChessboardCorners = board.chessboardCorners
                newChessboardCorners[:,1] = (7*1) - newChessboardCorners[:,1]
                for idx in idsLeft:
                    objpoints.append(newChessboardCorners[idx])
                objpoints = np.asarray(objpoints)
            # TODO : 검출된 charuco와 Id값 저장하기
        k = cv.waitKey(1)
        if k == 27:
            break
        elif k == 99: # C
            isCalibrating = not isCalibrating

if __name__ == "__main__":
    main()