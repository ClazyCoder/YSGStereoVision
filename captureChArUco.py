'''

'''
import cv2 as cv
import numpy as np
from cv2 import aruco
import cameramodule.cameramodule as cm
import os
import datetime
import json

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
    stereoCam = cm.StereoCamera(IMGSIZE)
    allCornersLeft = []
    allCornersRight = []
    allIdsLeft = []
    allIdsRight = []
    objpoints= []
    isCalibrating = False
    while True:
        leftFrame, rightFrame = stereoCam.get_frame()
        grayLeft = cv.cvtColor(leftFrame, cv.COLOR_BGR2GRAY)
        grayRight = cv.cvtColor(rightFrame, cv.COLOR_BGR2GRAY)
        img_left = leftFrame.copy()
        img_right = rightFrame.copy()
        if isCalibrating:
            cornersLeft, idsLeft, rejectedImgPointsLeft = cv.aruco.detectMarkers(grayLeft, aruco_dict)
            cornersRight, idsRight, rejectedImgPointsRight = cv.aruco.detectMarkers(grayRight, aruco_dict)
            if(len(cornersLeft) > 0 and len(cornersRight) > 0):
                for corner in cornersLeft:
                    cv.cornerSubPix(grayLeft, corner, (3,3), (-1,-1),criteria)
                    ret, charucoCorners, charucoIds = cv.aruco.interpolateCornersCharuco(cornersLeft,idsLeft,grayLeft,board)
                    if ret:
                        allCornersLeft.append(charucoCorners)
                        allIdsLeft.append(charucoIds)
                        cv.aruco.drawDetectedCornersCharuco(img_left, charucoCorners, charucoIds)
                for corner in cornersRight:
                    cv.cornerSubPix(grayRight,corner, (3,3),(-1,-1),criteria)
                    ret, charucoCorners, charucoIds = cv.aruco.interpolateCornersCharuco(cornersRight,idsRight,grayRight,board)
                    if ret:
                        allCornersRight.append(charucoCorners)
                        allIdsRight.append(charucoIds)
                        cv.aruco.drawDetectedCornersCharuco(img_right, charucoCorners, charucoIds)
                newChessboardCorners = board.chessboardCorners
                newChessboardCorners[:,1] = (7*1) - newChessboardCorners[:,1]
                for idx in idsLeft:
                    objpoints.append(newChessboardCorners[idx])
                objpoints = np.asarray(objpoints)
                
                today = datetime.datetime.today()
                filename_left = './ChAruco_captures/capture_left' + str(today.year)+str(today.month)+str(today.day)+'-'+str(today.hour)+"h"+str(today.minute)+"m"+str(today.second)+"s"+".jpg"
                jsonfilename_left = './ChAruco_datas/data_left' + str(today.year)+str(today.month)+str(today.day)+'-'+str(today.hour)+"h"+str(today.minute)+"m"+str(today.second)+"s"+".json"
                jsonfile_left = json.dumps(
                    {
                        "imgp" : allCornersLeft.tolist(),
                        "ids" : allIdsLeft.tolist()
                    }
                )
                with open(jsonfilename_left, 'w') as f:
                    f.write(jsonfile_left)
                filename_right = './ChAruco_captures/capture_right' + str(today.year)+str(today.month)+str(today.day)+'-'+str(today.hour)+"h"+str(today.minute)+"m"+str(today.second)+"s"+".jpg"
                jsonfilename_right = './ChAruco_datas/data_right' + str(today.year)+str(today.month)+str(today.day)+'-'+str(today.hour)+"h"+str(today.minute)+"m"+str(today.second)+"s"+".json"
                jsonfile_right = json.dumps(
                    {
                        "imgp" : allCornersRight.tolist(),
                        "ids" : allIdsRight.tolist()
                    }
                )
                with open(jsonfilename_right, 'w') as f:
                    f.write(jsonfile_right)
                cv.imwrite(filename_left, img_left)
                cv.imwrite(filename_right, img_right)
                cv.imshow('left', img_left)
                cv.imshow('right', img_right)
                cv.waitKey(3000)
            isCalibrating = False
        k = cv.waitKey(1)
        if k == 27:
            break
        elif k == 99: # C
            isCalibrating = not isCalibrating

if __name__ == "__main__":
    main()