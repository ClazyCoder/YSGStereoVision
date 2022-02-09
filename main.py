import cv2 as cv
import numpy as np
import datetime
import stereomodule.calibration as sc
import stereomodule.stereomatcher as ssm
import cameramodule.cameramodule as cm

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

    stereoCam = cm.StereoCamera(IMGSIZE)
    calibrator = sc.Calibrator(IMGSIZE)
    calibrator.load_calibration_datas()
    stereoMatcher = ssm.StereoMatcher(ssm.STEREO_TYPE_BM,NUM_DISPARITY,BLOCKSIZE)
    # stereoMatcher.leftMatcher.set...
    stereoMatcher.create_wls_filter()
    stereoMatcher.set_wls_filter_parameters(8000,1.5)

    K1, K2, D1, D2, R, T = calibrator.K1, calibrator.K2, calibrator.D1, calibrator.D2, calibrator.R, calibrator.T
    R1, R2, P1, P2, Q, roi_left, roi_right = cv.stereoRectify(K1, D1, K2, D2, IMGSIZE, R, T ,alpha=0,flags=cv.CALIB_ZERO_DISPARITY)
    leftMapX, leftMapY = cv.initUndistortRectifyMap(K1, D1, R1, P1, IMGSIZE, cv.CV_32FC1)
    rightMapX, rightMapY = cv.initUndistortRectifyMap(K2, D2, R2, P2, IMGSIZE, cv.CV_32FC1)
    valid_x1, valid_y1, valid_x2, valid_y2 = cv.getValidDisparityROI(roi_left,roi_right,minDisparity=0, numberOfDisparities=NUM_DISPARITY,blockSize=BLOCKSIZE)
    while True:
        leftImg, rightImg = stereoCam.get_frame()
        
        left_rectified = cv.remap(leftImg, leftMapX, leftMapY, cv.INTER_CUBIC, cv.BORDER_CONSTANT)
        right_rectified = cv.remap(rightImg, rightMapX, rightMapY, cv.INTER_CUBIC, cv.BORDER_CONSTANT)

        left_valid_rectified = left_rectified[valid_y1:valid_y1+valid_y2, valid_x1:valid_x1+valid_x2]
        right_valid_rectified = right_rectified[valid_y1:valid_y1+valid_y2, valid_x1:valid_x1+valid_x2]

        #gray_left = cv.cvtColor(left_valid_rectified, cv.COLOR_BGR2GRAY)
        #gray_right = cv.cvtColor(right_valid_rectified, cv.COLOR_BGR2GRAY)
        gray_left = cv.cvtColor(left_rectified, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(right_rectified, cv.COLOR_BGR2GRAY)

        disparity = stereoMatcher.get_disparity(gray_left, gray_right)
        filteredDisp = stereoMatcher.get_filtered_disparity(gray_left,gray_right)
        filteredDisp = filteredDisp.astype(np.float32)

        disparity_valid_filtered = filteredDisp[valid_y1:valid_y1+valid_y2, valid_x1:valid_x1+valid_x2]
        disparity_valid = disparity[valid_y1:valid_y1+valid_y2, valid_x1:valid_x1+valid_x2]
        image3D = cv.reprojectImageTo3D(disparity_valid_filtered, Q)
        depth = image3D[:,:,2].astype(np.float32)

        mask = disparity_valid_filtered > disparity_valid_filtered.min()

        cv.normalize(depth,depth,0.0,1.0,cv.NORM_MINMAX)
        cv.normalize(disparity_valid,disparity_valid,1.0,0,cv.NORM_MINMAX)
        cv.normalize(disparity_valid_filtered,disparity_valid_filtered,1.0,0,cv.NORM_MINMAX)
        cv.imshow('left',leftImg)
        cv.imshow('right',rightImg)
        cv.imshow('rectified_left',left_rectified)
        cv.imshow('rectified_right',right_rectified)
        #cv.imshow('rectified_left_valid',left_valid_rectified)
        #cv.imshow('rectified_right_valid',right_valid_rectified)
        cv.imshow('disparity',disparity_valid)
        cv.imshow('filteredDisparity',disparity_valid_filtered)

        k = cv.waitKey(1)

        if k == 27:
            break

        elif k == 99: # 'c'
            today = datetime.datetime.today()
            filename = './depth'+ str(today.year)+str(today.month)+str(today.day)+"-"+str(today.hour)+"h"+str(today.minute)+"m"+str(today.second)+"s"+".ply"
            depth_points = image3D[mask]
            point_colors = cv.cvtColor(left_valid_rectified,cv.COLOR_BGR2RGB)
            point_colors = point_colors[mask]
            point_colors = np.nan_to_num(point_colors,posinf=0,neginf=0)

            verts = depth_points.reshape(-1, 3)
            point_colors = point_colors.reshape(-1, 3)
            verts = np.hstack([verts, point_colors])
            with open(filename, 'w') as f:
                f.write(ply_header % dict(vert_num=len(verts)))
                np.savetxt(f, verts, '%f %f %f %d %d %d')
    
if __name__ == "__main__":
    main()