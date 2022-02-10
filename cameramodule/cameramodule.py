import cv2 as cv

class StereoCamera:
    def __init__(self,imgSize):
        self.leftCamera = cv.VideoCapture(0, cv.CAP_DSHOW)
        self.leftCamera.set(cv.CAP_PROP_AUTOFOCUS, 0)
        self.leftCamera.set(cv.CAP_PROP_AUTO_EXPOSURE,0)
        self.leftCamera.set(cv.CAP_PROP_AUTO_WB,0)
        self.leftCamera.set(cv.CAP_PROP_FRAME_WIDTH, imgSize[0])
        self.leftCamera.set(cv.CAP_PROP_FRAME_HEIGHT, imgSize[1])

        self.rightCamera = cv.VideoCapture(1, cv.CAP_DSHOW)
        self.rightCamera.set(cv.CAP_PROP_AUTOFOCUS, 0)
        self.rightCamera.set(cv.CAP_PROP_AUTO_EXPOSURE,0)
        self.rightCamera.set(cv.CAP_PROP_AUTO_WB,0)
        self.rightCamera.set(cv.CAP_PROP_FRAME_WIDTH, imgSize[0])
        self.rightCamera.set(cv.CAP_PROP_FRAME_HEIGHT, imgSize[1])
        
    def get_frame(self):
        if not (self.leftCamera.grab()and self.rightCamera.grab()):
            print("No more frames")
            return None
        ret_left, frame_left = self.leftCamera.retrieve()
        ret_right, frame_right = self.rightCamera.retrieve()
        return (frame_left, frame_right)
        
    def get_frame_with_ret(self):
        if not (self.leftCamera.grab()and self.rightCamera.grab()):
            print("No more frames")
            return None
        ret_left, frame_left = self.leftCamera.retrieve()
        ret_right, frame_right = self.rightCamera.retrieve()
        return (ret_left, frame_left, ret_right, frame_right)