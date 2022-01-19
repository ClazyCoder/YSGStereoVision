import cv2 as cv
import numpy as np
import StereoModule.Calibration as SC
import StereoModule.StereoMatcher as SSM
import CameraModule.CameraModule as CM

IMGSIZE = (640,480)

def main():
    stereoCam = CM.StereoCamera(IMGSIZE)
    calibrator = SC.Calibrator(IMGSIZE)
    calibrator.LoadCalibrationDatas()
    stereoMatcher = SSM.StereoMatcher(SSM.STEREO_TYPE_BM,112,15)
    # stereoMatcher.leftMatcher.set...
    stereoMatcher.CreateWlsFilter()

    K1, K2, D1, D2, R, T = calibrator.K1, calibrator.K2, calibrator.D1, calibrator.D2, calibrator.R, calibrator.T
    R1, R2, P1, P2, Q, roi_left, roi_right = cv.stereoRectify(K1, D1, K2, D2, IMGSIZE, R, T ,alpha=0,flags=cv.CALIB_ZERO_DISPARITY)
    leftMapX, leftMapY = cv.initUndistortRectifyMap(K1, D1, R1, P1, IMGSIZE, cv.CV_32FC1)
    rightMapX, rightMapY = cv.initUndistortRectifyMap(K2, D2, R2, P2, IMGSIZE, cv.CV_32FC1)

    while True:
        leftImg, rightImg = stereoCam.GetFrame()
        
        left_rectified = cv.remap(leftImg, leftMapX, leftMapY, cv.INTER_CUBIC, cv.BORDER_CONSTANT)
        right_rectified = cv.remap(rightImg, rightMapX, rightMapY, cv.INTER_CUBIC, cv.BORDER_CONSTANT)

        gray_left = cv.cvtColor(left_rectified, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(right_rectified, cv.COLOR_BGR2GRAY)

        disparity = stereoMatcher.GetDisparity(gray_left, gray_right)
        filteredDisp = stereoMatcher.GetFilteredDisparity(gray_left,gray_right)
        image3D = cv.reprojectImageTo3D(disparity, Q)
        depth = image3D[:,:,2].astype(np.float32)
        cv.normalize(depth,depth,0.0,1.0,cv.NORM_MINMAX)
        cv.normalize(disparity,disparity,1.0,0,cv.NORM_MINMAX)
        cv.normalize(filteredDisp,filteredDisp,255.0,0,cv.NORM_MINMAX)
        cv.imshow('left',leftImg)
        cv.imshow('right',rightImg)
        cv.imshow('rectified_left',left_rectified)
        cv.imshow('rectified_right',right_rectified)
        cv.imshow('disparity',disparity)
        cv.imshow('filteredDisparity',filteredDisp)
        cv.imshow('depth',depth)

        k = cv.waitKey(1)
        if k == 27:
            break
    

if __name__ == "__main__":
    main()