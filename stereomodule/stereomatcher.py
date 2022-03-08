import cv2 as cv
import numpy as np

STEREO_TYPE_BM = int(0)
STEREO_TYPE_SGBM = int(1)

class StereoMatcher:
    
    def __init__(self, stereo_type,num_disparities, block_size):
        if stereo_type == STEREO_TYPE_BM:
            self.stereo_type = STEREO_TYPE_BM
            self.left_matcher = cv.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
        elif stereo_type == STEREO_TYPE_SGBM:
            self.stereo_type = STEREO_TYPE_SGBM
            self.left_matcher = cv.StereoSGBM_create(numDisparities=num_disparities, blockSize=block_size)
        else:
            self.stereo_type = STEREO_TYPE_BM
            self.left_matcher = cv.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
        self.right_matcher = None
        self.wls_filter = None
    
    def set_min_disparities(self, min_disparities):
        self.left_matcher.setMinDisparities(min_disparities)

    def set_num_disparity(self, num_disparity):
        self.left_matcher.setNumDisparity(num_disparity)

    def set_block_size(self, block_size):
        self.left_matcher.setBlockSize(block_size)

    def set_pre_filter_cap(self, pre_filter_cap):
        self.left_matcher.setPreFilterCap(pre_filter_cap)

    def set_pre_filter_size(self, pre_filter_size):
        self.left_matcher.setPreFilterSize(pre_filter_size)

    def set_pre_filter_type(self, pre_filter_type):
        self.left_matcher.setPreFilterType(pre_filter_type)

    def set_uniqueness_ratio(self, uniqueness_ratio):
        self.left_matcher.setUniquenessRatio(uniqueness_ratio)

    def set_texture_threshold(self, texture_threshold):
        self.left_matcher.setTextureThreshold(texture_threshold)

    def set_speckle_range(self, speckle_range):
        self.left_matcher.setSpeckleRange(speckle_range)

    def set_speckle_window_size(self, speckle_window_size):
        self.left_matcher.setSpeckleWindowSize(speckle_window_size)

    def set_disp23_max_diff(self, disp23_max_diff):
        self.left_matcher.setDisp12MaxDiff(disp23_max_diff)

    def set_p1(self, p1):
        assert self.stereo_type == STEREO_TYPE_SGBM, 'Only SGBM has P1 parameter!'
        self.left_matcher.setP1(p1)

    def set_p2(self, p2):
        assert self.stereo_type == STEREO_TYPE_SGBM, 'Only SGBM has P2 parameter!'
        self.left_matcher.setP2(p2)

    def set_mode(self, mode):
        assert self.stereo_type == STEREO_TYPE_SGBM, 'Only SGBM has Matching Mode Parameter!'
        self.left_matcher.setMode(mode)

    def get_disparity(self, rectified_left, rectified_right):
        '''
        Disparity를 계산하여 반환하는 메서드
        '''
        disparity = self.left_matcher.compute(rectified_left, rectified_right)
        res = disparity.astype(np.float32) / 16.0
        return res

    def create_wls_filter(self):
        '''
        WlsFilter 객체를 생성하는 메서드
        '''
        self.right_matcher = cv.ximgproc.createRightMatcher(self.left_matcher)
        self.wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)

    def set_wls_filter_parameters(self, lmbda, sigma):
        '''
        WlsFilter의 패러미터를 변경하는 메서드
        '''
        self.wls_filter.setLambda(lmbda)
        self.wls_filter.setSigmaColor(sigma)

    def get_filtered_disparity(self, rectified_left, rectified_right):
        '''
        WlsFilter를 적용한 Disparity를 반환하는 메서드
        '''
        disp_left = self.left_matcher.compute(rectified_left, rectified_right)
        disp_right = self.right_matcher.compute(rectified_right,rectified_left)
        disp_left = disp_left.astype(np.float32) / 16.0
        disp_right = disp_right.astype(np.float32) / 16.0
        disp_left = np.int16(disp_left)
        disp_right = np.int16(disp_right)
        filtered_disp = self.wls_filter.filter(disp_left, rectified_left, None, disp_right)
        return filtered_disp
