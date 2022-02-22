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
    stereo_cam = cm.StereoCamera(IMGSIZE)
    all_corners_left = []
    all_corners_right = []
    all_ids_left = []
    all_ids_right = []
    obj_points= []
    is_calibrating = False
    while True:
        leftFrame, rightFrame = stereo_cam.get_frame()
        grayLeft = cv.cvtColor(leftFrame, cv.COLOR_BGR2GRAY)
        grayRight = cv.cvtColor(rightFrame, cv.COLOR_BGR2GRAY)
        img_left = leftFrame.copy()
        img_right = rightFrame.copy()
        if is_calibrating:
            corners_left, ids_left, rejectedImgPointsLeft = cv.aruco.detectMarkers(grayLeft, aruco_dict)
            corners_right, ids_right, rejectedImgPointsRight = cv.aruco.detectMarkers(grayRight, aruco_dict)
            if(len(corners_left) > 0 and len(corners_right) > 0):
                for corner in corners_left:
                    cv.cornerSubPix(grayLeft, corner, (3,3), (-1,-1),criteria)
                    ret, charucoCorners, charucoIds = cv.aruco.interpolateCornersCharuco(corners_left,ids_left,grayLeft,board)
                    if ret:
                        all_corners_left.append(charucoCorners)
                        all_ids_left.append(charucoIds)
                        cv.aruco.drawDetectedCornersCharuco(img_left, charucoCorners, charucoIds)
                for corner in corners_right:
                    cv.cornerSubPix(grayRight,corner, (3,3),(-1,-1),criteria)
                    ret, charucoCorners, charucoIds = cv.aruco.interpolateCornersCharuco(corners_right,ids_right,grayRight,board)
                    if ret:
                        all_corners_right.append(charucoCorners)
                        all_ids_right.append(charucoIds)
                        cv.aruco.drawDetectedCornersCharuco(img_right, charucoCorners, charucoIds)
                newChessboardCorners = board.chessboardCorners
                newChessboardCorners[:,1] = (7*1) - newChessboardCorners[:,1]
                for idx in ids_left:
                    obj_points.append(newChessboardCorners[idx])
                obj_points = np.asarray(obj_points)
                
                today = datetime.datetime.today()
                filename_left = './ChAruco_captures/capture_left' + str(today.year)+str(today.month)+str(today.day)+'-'+str(today.hour)+"h"+str(today.minute)+"m"+str(today.second)+"s"+".jpg"
                jsonfilename_left = './ChAruco_datas/data_left' + str(today.year)+str(today.month)+str(today.day)+'-'+str(today.hour)+"h"+str(today.minute)+"m"+str(today.second)+"s"+".json"
                jsonfile_left = json.dumps(
                    {
                        "imgp" : all_corners_left.tolist(),
                        "ids" : all_ids_left.tolist()
                    }
                )
                with open(jsonfilename_left, 'w') as f:
                    f.write(jsonfile_left)
                filename_right = './ChAruco_captures/capture_right' + str(today.year)+str(today.month)+str(today.day)+'-'+str(today.hour)+"h"+str(today.minute)+"m"+str(today.second)+"s"+".jpg"
                jsonfilename_right = './ChAruco_datas/data_right' + str(today.year)+str(today.month)+str(today.day)+'-'+str(today.hour)+"h"+str(today.minute)+"m"+str(today.second)+"s"+".json"
                jsonfile_right = json.dumps(
                    {
                        "imgp" : all_corners_right.tolist(),
                        "ids" : all_ids_right.tolist()
                    }
                )
                with open(jsonfilename_right, 'w') as f:
                    f.write(jsonfile_right)
                cv.imwrite(filename_left, img_left)
                cv.imwrite(filename_right, img_right)
                cv.imshow('left', img_left)
                cv.imshow('right', img_right)
                cv.waitKey(3000)
            is_calibrating = False
        k = cv.waitKey(1)
        if k == 27:
            break
        elif k == 99: # C
            is_calibrating = not is_calibrating

if __name__ == "__main__":
    main()