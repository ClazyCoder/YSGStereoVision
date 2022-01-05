import cv2 as cv
import numpy as np

STEREO_TYPE_BM = int(0)
STEREO_TYPE_SGBM = int(1)

class StereoMatcher:
    def __init__(self, windowSize, numDisparity, stereoType):
        # TODO : 스테레오 모듈 패러미터 초기화
        if stereoType == STEREO_TYPE_BM:
            pass
        elif stereoType == STEREO_TYPE_SGBM:
            pass
        else:
            pass
        pass
    def getDisparity(self, rectifiedImg1, rectifiedImg2):
        # TODO : 디스패러티 계산
        pass