import cv2 as cv
import numpy as np
import json
import glob

class Calibrator:
    def __init__(self, imgWidth, imgHeight):
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight
        self.leftDatas = []
        self.rightDats = []
        pass

    def LoadDatas(self, path='./datas'):
        leftdata_glob = glob.glob(path+'/data_left*.json')
        rightdata_glob = glob.glob(path+'/data_left*.json')
        # TODO: 데이터 불러오기
        pass

    def RunCalibration(self):
        # TODO: 캘리브레이션 수행
        pass