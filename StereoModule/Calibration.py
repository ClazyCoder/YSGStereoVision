import cv2 as cv
import numpy as np
import json
import glob
import os

class Calibrator:

    def __init__(self, imgSize):
        self.imgSize = imgSize
        self.objDatas = []
        self.leftDatas = []
        self.rightDatas = []
        self.K1 = None
        self.K2 = None
        self.D1 = None
        self.D2 = None
        self.R  = None
        self.T  = None
        self.E  = None
        self.F  = None
        self.isCalibrated = False

    def LoadDatas(self, path='./datas'):
        leftdata_glob = glob.glob(path+'/data_left*.json')
        rightdata_glob = glob.glob(path+'/data_left*.json')

        for data in leftdata_glob:
            with open(data,'r') as f:
                fstr = f.read()
                f.close()
            jstr = json.loads(fstr)
            objPoints = jstr['objp']
            imgPoints = jstr['imgp']
            objPoints, imgPoints = np.array(objPoints, dtype=np.float32), np.array(imgPoints, dtype=np.float32)
            self.objDatas.append(objPoints)
            self.leftDatas.append(imgPoints)

        for data in rightdata_glob:
            with open(data,'r') as f:
                fstr = f.read()
                f.close()
            jstr = json.loads(fstr)
            imgPoints = jstr['imgp']
            imgPoints = np.array(imgPoints, dtype=np.float32)
            self.rightDatas.append(imgPoints)

    def RunCalibration(self):
        _, mtx1, dist1, rvecs1, tvecs1 =cv.calibrateCamera(self.objDatas, self.leftDatas, self.imgSize, None, None)
        _, mtx2, dist2, rvecs2, tvecs2 =cv.calibrateCamera(self.objDatas, self.rightDatas, self.imgSize, None, None)
        
        ret, self.K1, self.D1,  \
        self.K2, self.D2,       \
        self.R, self.T,         \
        self.E, self.F = cv.stereoCalibrate(
             self.objDatas, self.leftDatas, self.rightDatas,
              mtx1, dist1, mtx2, dist2, self.imgSize, 
              flags=cv.CALIB_USE_INTRINSIC_GUESS+cv.CALIB_FIX_FOCAL_LENGTH)
        self.isCalibrated = True

    def SaveCalibrationDatas(self, directory='./calib'):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        filename = 'calib_data.json'
        jsonfile_calib = json.dumps(
            {
                "K1" : self.K1.tolist(),
                "K2" : self.K2.tolist(),
                "D1" : self.D1.tolist(),
                "D2" : self.D2.tolist(),
                "R" : self.R.tolist(),
                "T" : self.T.tolist(),
                "E" : self.E.tolist(),
                "F" : self.F.tolist(),
                "imgSize" : self.imgSize 
            }
        )
        with open(directory+'/'+filename, 'w') as f:
            f.write(jsonfile_calib)
    
    def LoadCalibrationDatas(self, filename='./calib/calib_data.json'):
        with open(filename,'r') as f:
            fstr = f.read()
        f.close()
        jstr = json.loads(fstr)
        self.K1 = np.array(jstr["K1"])
        self.K2 = np.array(jstr["K2"])
        self.D1 = np.array(jstr["D1"])
        self.D2 = np.array(jstr["D2"])
        self.R = np.array(jstr["R"])
        self.T = np.array(jstr["T"])
        self.isCalibrated = True

def main():
    calibrator = Calibrator((640,480))
    calibrator.LoadDatas('./datas')
    calibrator.RunCalibration()
    calibrator.SaveCalibrationDatas('./calib')

if __name__ == "__main__":
    main()