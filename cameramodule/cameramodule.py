import cv2 as cv

class StereoCamera:
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
        if not (self.left_camera.grab()and self.right_camera.grab()):
            print("No more frames")
            return None
        ret_left, frame_left = self.left_camera.retrieve()
        ret_right, frame_right = self.right_camera.retrieve()
        return (frame_left, frame_right)
        
    def get_frame_with_ret(self):
        if not (self.left_camera.grab()and self.right_camera.grab()):
            print("No more frames")
            return None
        ret_left, frame_left = self.left_camera.retrieve()
        ret_right, frame_right = self.right_camera.retrieve()
        return (ret_left, frame_left, ret_right, frame_right)