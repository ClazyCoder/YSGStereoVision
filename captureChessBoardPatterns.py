import cv2 as cv
import numpy as np
import json
import os
import datetime
import CameraModule.CameraModule as CM

IMGSIZE = (640,480)

def main():
    if not os.path.isdir('./captures'):
        os.mkdir('./captures')

    if not os.path.isdir('./datas'):
        os.mkdir('./datas')
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((7*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    stereoCam = CM.StereoCamera(IMGSIZE)

    isCalibrating = False

    while True:
        ret_left, leftImg, ret_right, rightImg = stereoCam.GetFrameWithRet()
        grayLeft = cv.cvtColor(leftImg, cv.COLOR_BGR2GRAY)
        grayRight = cv.cvtColor(rightImg,cv.COLOR_BGR2GRAY)
        if isCalibrating:
            retLeftCorner, corners_left = cv.findChessboardCorners(grayLeft, (9,7), flags=cv.CALIB_CB_ADAPTIVE_THRESH)
            retRightCorner, corners_right = cv.findChessboardCorners(grayRight, (9,7), flags=cv.CALIB_CB_ADAPTIVE_THRESH)
            if retLeftCorner and retRightCorner:
                objpoints.append(objp)
                corners_left2 = cv.cornerSubPix(grayLeft,corners_left, (11,11), (-1,-1), criteria)
                corners_right2 = cv.cornerSubPix(grayRight,corners_right, (11,11), (-1,-1), criteria)
                imgpoints_left.append(corners_left2)
                imgpoints_right.append(corners_right2)

                img_left = leftImg.copy()
                img_right = rightImg.copy()
                cv.drawChessboardCorners(img_left, (9,7), corners_left2, ret_left)
                cv.drawChessboardCorners(img_right, (9,7), corners_right2, ret_right)
                today = datetime.datetime.today()
                filename_left = './captures/capture_left' + str(today.year)+str(today.month)+str(today.day)+'-'+str(today.hour)+"h"+str(today.minute)+"m"+str(today.second)+"s"+".jpg"
                jsonfilename_left = './datas/data_left' + str(today.year)+str(today.month)+str(today.day)+'-'+str(today.hour)+"h"+str(today.minute)+"m"+str(today.second)+"s"+".json"
                jsonfile_left = json.dumps(
                    {
                        "objp" : objp.tolist(),
                        "imgp" : corners_left2.tolist()
                    }
                )
                with open(jsonfilename_left, 'w') as f:
                    f.write(jsonfile_left)
                filename_right = './captures/capture_right' + str(today.year)+str(today.month)+str(today.day)+'-'+str(today.hour)+"h"+str(today.minute)+"m"+str(today.second)+"s"+".jpg"
                jsonfilename_right = './datas/data_right' + str(today.year)+str(today.month)+str(today.day)+'-'+str(today.hour)+"h"+str(today.minute)+"m"+str(today.second)+"s"+".json"
                jsonfile_right = json.dumps(
                    {
                        "objp" : objp.tolist(),
                        "imgp" : corners_right2.tolist()
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
        cv.imshow('left',leftImg)
        cv.imshow('right',rightImg)
        k = cv.waitKey(1)
        if k == 27:
            break
        elif k == 99: # C
            isCalibrating = not isCalibrating
        
if __name__ == "__main__":
    main()