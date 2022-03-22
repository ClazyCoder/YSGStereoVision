import cv2 as cv
import numpy as np

STEREO_TYPE_BM = int(0)
STEREO_TYPE_SGBM = int(1)

class StereoMatcher:
    '''
    # StereoMatcher 클래스
    정렬된(Rectified) 두 영상을 입력받아 Stereo Matching을 수행한다.
    
    ## Init Parameters:
    - stereo_type
    - num_disparities
    - block_size

    ### stereo_type
    Block Matching 방법을 결정하는 패러미터.\n
    stereomatcher.STEREO_TYPE_BM : 일반적인 BM 모델.\n
    stereomatcher.STEREO_TYPE_SGBM : Semi-Global Block Matching 모델.

    ### num_disparities
    Block Matching시 검출할 Disparity의 단계를 결정하는 패러미터. 16의 배수여야 함.\n
    16의 배수인 정수 값.

    ### block_size
    Block Matching시 Block의 크기를 결정하는 패러미터\n
    홀수인 정수 값.
    '''
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
    
    def set_min_disparity(self, min_disparity):
        '''
        #set_min_disparity
        BM과 SGBM 클래스의 MinDisparity 패러미터를 설정하는 메서드
        
        ## Parameters
        - min_disparity

        ### min_disparity
        Matching시 최소 disparity의 기준값
        '''
        self.left_matcher.setMinDisparity(min_disparity)

    def set_num_disparities(self, num_disparities):
        '''
        # set_num_disparities
        BM과 SGBM 클래스의 NumDisparities 패러미터를 설정하는 메서드

        ## Parameters
        - num_disparities

        ### num_disparities
        Matching시 Disparity의 탐색 범위. (16의 배수여야 함, MaxDisparity - MinDisparity값.)
        '''
        self.left_matcher.setNumDisparities(num_disparities)

    def set_block_size(self, block_size):
        '''
        # set_block_size
        BM과 SGBM 클래스의 BlockSize 패러미터를 설정하는 메서드

        ## Parameters
        - block_size

        ### block_size
        Matching시의 비교할 Block의 크기.
        '''
        self.left_matcher.setBlockSize(block_size)

    def set_pre_filter_cap(self, pre_filter_cap):
        '''
        # set_pre_filter_cap
        BM과 SGBM 클래스의 PreFilterCap 패러미터를 설정하는 메서드

        ## Parameters
        - pre_filter_cap

        ### pre_filter_cap
        사전 필터링 이후 절단값. x축 기준으로 영상의 미분값을 계산하고,\n
        [-preFilterCap, preFilterCap]간격으로 값을 자른다.
        '''
        self.left_matcher.setPreFilterCap(pre_filter_cap)

    def set_pre_filter_size(self, pre_filter_size):
        '''
        # set_pre_filter_size
        BM 클래스의 PreFilterSize 패러미터를 설정하는 메서드

        ## Parameters
        - pre_filter_size

        ### pre_filter_size
        사전 필터링시 필터의 크기
        '''
        assert self.stereo_type == STEREO_TYPE_BM, 'Only BM has PreFilterSize parameter!'
        self.left_matcher.setPreFilterSize(pre_filter_size)

    def set_pre_filter_type(self, pre_filter_type):
        '''
        # set_pre_filter_type
        BM 클래스의 PreFilterType 패러미터를 설정하는 메서드

        ## Parameters
        - pre_filter_type

        ### pre_filter_type
        사전 필터링의 종류
        '''
        assert self.stereo_type == STEREO_TYPE_BM, 'Only BM has PreFilterType parameter!'
        self.left_matcher.setPreFilterType(pre_filter_type)

    def set_uniqueness_ratio(self, uniqueness_ratio):
        '''
        # set_uniqueness_ratio
        BM과 SGBM 클래스의 UniquenessRatio 패러미터를 설정하는 메서드

        ## Parameters
        - uniqueness_ratio

        ## uniqueness_ratio
        매칭시 비용함수에서 가장 높은 점수를 얻는지점(가장 확실하다고 보이는 disparity)이 없으면,\n
        그때 1등과 2등간의 임계점을 결정해주는 패러미터이다.
        '''
        self.left_matcher.setUniquenessRatio(uniqueness_ratio)

    def set_texture_threshold(self, texture_threshold):
        '''
        # set_texture_threshold
        BM 클래스의 TextureThreshold 패러미터를 설정하는 메서드

        ## Parameters
        - texture_threshold

        ### texture_threshold
        블록 매칭시 이 임계값 보다 낮은 텍스처 값(이미지 미분의 절대합)을\n
        지니는 지점은 disparity가 결정되지 않는다.
        '''
        assert self.stereo_type == STEREO_TYPE_BM, 'Only BM has TextureThreshold parameter!'
        self.left_matcher.setTextureThreshold(texture_threshold)
    
    def set_speckle_range(self, speckle_range):
        '''
        # set_speckle_range
        BM과 SGBM 클래스의 SpeckleRange 패러미터를 설정하는 메서드

        ## Parameters
        - speckle_range

        ### speckle_range
        블록 매칭시 주변 Disparity들에 비해 튀는 값을 가지는 작은 블롭(Blob)들을 제거하는 후처리 필터링 과정에 사용되는 패러미터이다.\n
        Blob과 그 주변 픽셀이 같은 덩어리인지를 결정할때 기준이 되는 범위 값이다. 
        '''
        self.left_matcher.setSpeckleRange(speckle_range)

    def set_speckle_window_size(self, speckle_window_size):
        '''
        # set_peckle_window_size
        BM과 SGBM 클래스의 SpeckleWindowSize 패러미터를 설정하는 메서드

        ## Parameters
        - speckle_window_size

        ### speckle_window_size
        블록 매칭시 주변 Disparity들에 비해 튀는 값을 가지는 작은 블롭(Blob)들을 제거하는 후처리 필터링 과정에 사용되는 패러미터이다.\n
        Blob의 크기를 결정하는 패러미터 이다.
        '''
        self.left_matcher.setSpeckleWindowSize(speckle_window_size)

    def set_p1(self, p1):
        '''
        # set_p1
        SGBM 클래스의 P1 패러미터를 설정하는 메서드

        ## Parameters
        - p1

        ### p1
        SGM(Semi-Global Matching)에 사용되는 패러미터이다.\n
        주변 픽셀의 수가 하나일때 disparity 차이에 대한 패널티 값이다.
        '''
        assert self.stereo_type == STEREO_TYPE_SGBM, 'Only SGBM has P1 parameter!'
        self.left_matcher.setP1(p1)

    def set_p2(self, p2):
        '''
        # set_p2
        SGBM 클래스의 P2 패러미터를 설정하는 메서드

        ## Parameters
        - p2

        ### p2
        SGM(Semi-Global Matching)에 사용되는 패러미터이다.\n
        주변 픽셀의 수가 둘 이상일때 disparity 차이에 대한 패널티 값이다.
        '''
        assert self.stereo_type == STEREO_TYPE_SGBM, 'Only SGBM has P2 parameter!'
        self.left_matcher.setP2(p2)

    def set_disp12_max_diff(self, disp12_max_diff):
        '''
        # set_disp12_max_diff
        SGBM 클래스의 Disp12MaxDiff 패러미터를 설정하는 메서드

        ## Parameters
        - disp12_max_diff

        ### disp12_max_diff
        왼쪽에서 오른쪽으로의 disparity 계산결과와 오른쪽에서 왼쪽으로의 계산결과를 비교하여,\n 
        이 값보다 차이가 클 경우, 그 픽셀의 disparity는 결정되지 않도록하는 패러미터이다.
        '''
        assert self.stereo_type == STEREO_TYPE_SGBM, 'Only SGBM has Disp12MaxDiff parameter!'
        self.left_matcher.setDisp12MaxDiff(disp12_max_diff)

    def set_mode(self, mode):
        '''
        # set_mode
        SGBM 클래스의 Mode 패러미터를 설정하는 메서드

        ## Parameters
        - mode

        ### mode
        SGBM의 계산 방식을 결정하는 패러미터이다.\n
        모드의 종류는 다음과 같다.
        - cv2.STEREO_SGBM_MODE_SGBM (기본값)
        - cv2.STEREO_SGBM_MODE_SGBM_3WAY
        - cv2.STEREO_SGBM_MODE_HH
        - cv2.STEREO_SGBM_MODE_HH4
        '''
        assert self.stereo_type == STEREO_TYPE_SGBM, 'Only SGBM has Matching Mode Parameter!'
        self.left_matcher.setMode(mode)

    def get_disparity(self, rectified_left, rectified_right):
        '''
        # get_disparity
        Disparity를 계산하여 반환하는 메서드

        ## Parameters
        - rectified_left
        - rectified_right

        ### rectified_left
        rectification이 끝난 왼쪽 이미지

        ### rectified_right
        rectification이 끝난 오른쪽 이미지
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
