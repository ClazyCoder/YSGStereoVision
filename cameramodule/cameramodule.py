import cv2 as cv

class StereoCamera:
    '''
    # StereoCamera 클래스
    스테레오 카메라 클래스.\n
    두 대의 카메라를 관리하고 영상을 받아온다.

    ## Init Parameters:
    - img_size

    ### img_size
    카메라의 해상도를 결정하는 패러미터 (WIDTH, HEIGHT)
    '''
    def __init__(self,img_size):
        self.left_camera = cv.VideoCapture(0, cv.CAP_DSHOW)
        self.left_camera.set(cv.CAP_PROP_AUTOFOCUS, 0)
        self.left_camera.set(cv.CAP_PROP_AUTO_EXPOSURE,0)
        self.left_camera.set(cv.CAP_PROP_AUTO_WB,0)
        self.left_camera.set(cv.CAP_PROP_FRAME_WIDTH, img_size[0])
        self.left_camera.set(cv.CAP_PROP_FRAME_HEIGHT, img_size[1])

        self.right_camera = cv.VideoCapture(1, cv.CAP_DSHOW)
        self.right_camera.set(cv.CAP_PROP_AUTOFOCUS, 0)
        self.right_camera.set(cv.CAP_PROP_AUTO_EXPOSURE,0)
        self.right_camera.set(cv.CAP_PROP_AUTO_WB,0)
        self.right_camera.set(cv.CAP_PROP_FRAME_WIDTH, img_size[0])
        self.right_camera.set(cv.CAP_PROP_FRAME_HEIGHT, img_size[1])
        
    def get_frame(self):
        '''
        두 카메라로 부터 얻은 영상을 반환하는 메서드
        '''
        if not (self.left_camera.grab()and self.right_camera.grab()):
            print("No more frames")
            return None
        ret_left, frame_left = self.left_camera.retrieve()
        ret_right, frame_right = self.right_camera.retrieve()
        return (frame_left, frame_right)
        
    def get_frame_with_ret(self):
        '''
        get_frame 메서드에서 retrieve 값을 추가적으로 반환하는 메서드
        '''
        if not (self.left_camera.grab()and self.right_camera.grab()):
            print("No more frames")
            return None
        ret_left, frame_left = self.left_camera.retrieve()
        ret_right, frame_right = self.right_camera.retrieve()
        return (ret_left, frame_left, ret_right, frame_right)