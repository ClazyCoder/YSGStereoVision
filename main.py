import cv2 as cv
import numpy as np
import datetime
import StereoModule.Calibration as SC
import StereoModule.StereoMatcher as SSM
import CameraModule.CameraModule as CM

IMGSIZE = (640,480)

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def main():
    NUM_DISPARITY = 112
    BLOCKSIZE = 15

    stereoCam = CM.StereoCamera(IMGSIZE)
    calibrator = SC.Calibrator(IMGSIZE)
    calibrator.LoadCalibrationDatas()
    stereoMatcher = SSM.StereoMatcher(SSM.STEREO_TYPE_BM,NUM_DISPARITY,BLOCKSIZE)
    # stereoMatcher.leftMatcher.set...
    stereoMatcher.CreateWlsFilter()
    stereoMatcher.SetWlsFilterParameters(8000,1.5)

    K1, K2, D1, D2, R, T = calibrator.K1, calibrator.K2, calibrator.D1, calibrator.D2, calibrator.R, calibrator.T
    R1, R2, P1, P2, Q, roi_left, roi_right = cv.stereoRectify(K1, D1, K2, D2, IMGSIZE, R, T ,alpha=0,flags=cv.CALIB_ZERO_DISPARITY)
    leftMapX, leftMapY = cv.initUndistortRectifyMap(K1, D1, R1, P1, IMGSIZE, cv.CV_32FC1)
    rightMapX, rightMapY = cv.initUndistortRectifyMap(K2, D2, R2, P2, IMGSIZE, cv.CV_32FC1)
    valid_x1, valid_y1, valid_x2, valid_y2 = cv.getValidDisparityROI(roi_left,roi_right,numberOfDisparities=NUM_DISPARITY,blockSize=BLOCKSIZE)
    while True:
        leftImg, rightImg = stereoCam.GetFrame()
        
        left_rectified = cv.remap(leftImg, leftMapX, leftMapY, cv.INTER_CUBIC, cv.BORDER_CONSTANT)
        right_rectified = cv.remap(rightImg, rightMapX, rightMapY, cv.INTER_CUBIC, cv.BORDER_CONSTANT)

        gray_left = cv.cvtColor(left_rectified, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(right_rectified, cv.COLOR_BGR2GRAY)

        disparity = stereoMatcher.GetDisparity(gray_left, gray_right)
        filteredDisp = stereoMatcher.GetFilteredDisparity(gray_left,gray_right)
        filteredDisp = filteredDisp.astype(np.float32)
        left_valid_rectified = left_rectified[valid_y1:valid_y1+valid_y2, valid_x1:valid_x1+valid_x2]
        right_valid_rectified = right_rectified[valid_y1:valid_y1+valid_y2, valid_x1:valid_x1+valid_x2]
        disparity_valid = filteredDisp[valid_y1:valid_y1+valid_y2, valid_x1:valid_x1+valid_x2]

        image3D = cv.reprojectImageTo3D(disparity_valid, Q)
        depth = image3D[:,:,2].astype(np.float32)


        mask = disparity_valid > disparity_valid.min()

        cv.normalize(depth,depth,0.0,1.0,cv.NORM_MINMAX)
        cv.normalize(disparity,disparity,1.0,0,cv.NORM_MINMAX)
        cv.normalize(filteredDisp,filteredDisp,1.0,0,cv.NORM_MINMAX)
        cv.imshow('left',leftImg)
        cv.imshow('right',rightImg)
        #cv.imshow('rectified_left',left_rectified)
        #cv.imshow('rectified_right',right_rectified)
        cv.imshow('rectified_left_valid',left_valid_rectified)
        cv.imshow('rectified_right_valid',right_valid_rectified)
        cv.imshow('disparity',disparity)
        cv.imshow('filteredDisparity',filteredDisp)

        k = cv.waitKey(1)

        if k == 27:
            break

        elif k == 99: # 'c'
            today = datetime.datetime.today()
            filename = './depth'+ str(today.year)+str(today.month)+str(today.day)+"-"+str(today.hour)+"시"+str(today.minute)+"분"+str(today.second)+"초"+".ply"
            depth_points = depth[mask]
            point_colors = left_valid_rectified[mask]

            verts = depth_points.reshape(-1, 3)
            point_colors = point_colors.reshape(-1, 3)
            verts = np.hstack([verts, point_colors])
            with open(filename, 'w') as f:
                f.write(ply_header % dict(vert_num=len(verts)))
                np.savetxt(f, verts, '%f %f %f %d %d %d')
    

if __name__ == "__main__":
    main()