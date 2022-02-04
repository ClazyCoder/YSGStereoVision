import cv2 as cv
from cv2 import aruco
import numpy as np
import json
import glob
import os

class Calibrator:

    def __init__(self, imgSize):
        self.imgSize = imgSize[::-1]
        self.objDatas = []
        self.leftDatas = []
        self.rightDatas = []
        self.leftChDatas = []
        self.rightChDatas = []
        self.leftChIds = []
        self.rightChIds = []
        self.K1 = None
        self.K2 = None
        self.D1 = None
        self.D2 = None
        self.R  = None
        self.T  = None
        self.E  = None
        self.F  = None
        self.isCalibrated = False
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.Charucoboard = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)

    def LoadDatas(self, path='./datas'):
        leftdata_glob = glob.glob(path+'/data_left*.json')
        rightdata_glob = glob.glob(path+'/data_right*.json')

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

    def LoadChArUcoDatas(self, path='./ChAruco_datas'):
        leftdata_glob = glob.glob(path+'/data_left*.json')
        rightdata_glob = glob.glob(path+'/data_right*.json')

        for data in leftdata_glob:
            with open(data,'r') as f:
                fstr = f.read()
                f.close()
            jstr = json.loads(fstr)
            imgPoints = jstr['imgp']
            Ids = jstr['ids']
            Ids, imgPoints = np.array(Ids, dtype=np.float32), np.array(imgPoints, dtype=np.float32)
            self.leftChDatas.append(imgPoints)
            self.leftChIds.append(Ids)
        for data in rightdata_glob:
            with open(data,'r') as f:
                fstr = f.read()
                f.close()
            jstr = json.loads(fstr)
            imgPoints = jstr['imgp']
            Ids = jstr['ids']
            Ids, imgPoints = np.array(Ids, dtype=np.float32), np.array(imgPoints, dtype=np.float32)
            self.rightChDatas.append(imgPoints)
            self.rightChIds.append(Ids)

    def RunCalibration(self):
        _, mtx1, dist1, rvecs1, tvecs1 =cv.calibrateCamera(self.objDatas, self.leftDatas, self.imgSize, None, None)
        _, mtx2, dist2, rvecs2, tvecs2 =cv.calibrateCamera(self.objDatas, self.rightDatas, self.imgSize, None, None)
        
        ret, self.K1, self.D1,  \
        self.K2, self.D2,       \
        self.R, self.T,         \
        self.E, self.F= cv.stereoCalibrate(
             self.objDatas, self.leftDatas, self.rightDatas,
              mtx1, dist1, mtx2, dist2, self.imgSize,
              flags=cv.CALIB_FIX_INTRINSIC)
        self.isCalibrated = True
        mean_error = 0
        for i in range(len(self.objDatas)):
            imgpoints_left, _ = cv.projectPoints(self.objDatas[i], rvecs1[i], tvecs1[i], mtx1, dist1)
            error = cv.norm(self.leftDatas[i], imgpoints_left, cv.NORM_L2)/len(imgpoints_left)
            mean_error += error
        print( "total error for left: {}".format(mean_error/len(self.objDatas)) )
        mean_error = 0
        for i in range(len(self.objDatas)):
            imgpoints_right, _ = cv.projectPoints(self.objDatas[i], rvecs2[i], tvecs2[i], mtx2, dist2)
            error = cv.norm(self.rightDatas[i], imgpoints_right, cv.NORM_L2)/len(imgpoints_right)
            mean_error += error
        print( "total error for right: {}".format(mean_error/len(self.objDatas)) )
    
    def RunCalibrationWithChArUco(self):
        _, mtx1, dist1, rvecs1, tvecs1, _, _ = cv.aruco.calibrateCameraCharuco(
                      charucoCorners=self.leftChDatas,
                      charucoIds=self.leftChIds,
                      board=self.Charucoboard,
                      imageSize=self.imgSize,
                      cameraMatrix=None,
                      distCoeffs=None,
                      flags=cv.CALIB_USE_INTRINSIC_GUESS,
                      criteria=(cv.TERM_CRITERIA_EPS & cv.TERM_CRITERIA_COUNT, 10000, 1e-9))
        _, mtx2, dist2, rvecs2, tvecs2, _, _ = cv.aruco.calibrateCameraCharuco(
                      charucoCorners=self.rightChDatas,
                      charucoIds=self.rightChIds,
                      board=self.Charucoboard,
                      imageSize=self.imgSize,
                      cameraMatrix=None,
                      distCoeffs=None,
                      flags=cv.CALIB_USE_INTRINSIC_GUESS,
                      criteria=(cv.TERM_CRITERIA_EPS & cv.TERM_CRITERIA_COUNT, 10000, 1e-9))
        ret, self.K1, self.D1,  \
        self.K2, self.D2,       \
        self.R, self.T,         \
        self.E, self.F= cv.stereoCalibrate(
             self.objDatas, self.leftDatas, self.rightDatas,
              mtx1, dist1, mtx2, dist2, self.imgSize,
              flags=cv.CALIB_FIX_INTRINSIC)
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
                "imgSize" : self.imgSize[::-1] 
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