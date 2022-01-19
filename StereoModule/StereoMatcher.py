import cv2 as cv
import numpy as np

STEREO_TYPE_BM = int(0)
STEREO_TYPE_SGBM = int(1)

class StereoMatcher:
    
    def __init__(self, stereoType,numDisparities, blockSize):
        if stereoType == STEREO_TYPE_BM:
            self.leftMatcher = cv.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
        elif stereoType == STEREO_TYPE_SGBM:
            self.leftMatcher = cv.StereoSGBM_create(numDisparities=numDisparities, blockSize=blockSize)
        else:
            self.leftMatcher = cv.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
        self.rightMatcher = None
        self.wlsFilter = None
        
        
    def GetDisparity(self, rectifiedLeft, rectifiedRight):
        disparity = self.leftMatcher.compute(rectifiedLeft, rectifiedRight)
        res = disparity.astype(np.float32) / 16.0
        return res

    def CreateWlsFilter(self):
        self.rightMatcher = cv.ximgproc.createRightMatcher(self.leftMatcher)
        self.wlsFilter = cv.ximgproc.createDisparityWLSFilter(matcher_left=self.leftMatcher)

    def SetWlsFilterParameters(self, lmbda, sigma):
        self.wlsFilter.setLambda(lmbda)
        self.wlsFilter.setSigmaColor(sigma)

    def GetFilteredDisparity(self, rectifiedLeft, rectifiedRight):
        dispLeft = self.leftMatcher.compute(rectifiedLeft, rectifiedRight).astype(np.float32) / 16.0
        dispRight = self.rightMatcher.compute(rectifiedLeft, rectifiedRight).astype(np.float32) / 16.0
        dispLeft = np.int16(dispLeft)
        dispRight = np.int16(dispRight)
        filteredDisp = self.wlsFilter.filter(dispLeft, rectifiedLeft, None, dispRight)
        return filteredDisp
