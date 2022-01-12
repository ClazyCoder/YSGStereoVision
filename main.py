import cv2 as cv
import numpy as np
import StereoModule.Calibration as SC
import StereoModule.StereoMatcher as SSM
import CameraModule.CameraModule as CM

def main():
    StereoCam = CM.StereoCamera((640,480))
    while True:
        left, right = StereoCam.GetFrame()
        # TODO : Depth값 추정
        k = cv.waitKey(1)
        if k == 99:
            break
    

if __name__ == "__main__":
    main()