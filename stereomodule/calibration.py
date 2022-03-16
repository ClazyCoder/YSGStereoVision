import cv2 as cv
from cv2 import aruco
import numpy as np
import json
import glob
import os

class Calibrator:
    '''
    # Calibrator 클래스
    패턴을 캡쳐한 영상위의 좌표와 실제 패턴의 월드 좌표를 사용하여 캘리브레이션을 수행하고,
    카메라 메트릭스를 얻어내는 클래스

    ## Init Parameters:
    - img_size

    ### img_size
    캘리브레이션시 사용될 패턴이미지의 사이즈 (WIDTH, HEIGHT)
    '''
    def __init__(self, img_size):
        self.img_size = img_size[::-1]
        self.obj_datas = []
        self.left_datas = []
        self.right_datas = []
        self.left_Ch_datas = []
        self.right_Ch_datas = []
        self.left_Ch_ids = []
        self.right_Ch_ids = []
        self.K1 = None
        self.K2 = None
        self.D1 = None
        self.D2 = None
        self.R  = None
        self.T  = None
        self.E  = None
        self.F  = None
        self.is_calibrated = False
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.Charuco_board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)

    def load_datas(self, path='./datas'):
        '''
        # load_datas
        지정된 path로 부터 패턴 데이터를 불러오는 메서드

        ## Parameters
        - path

        ### path
        데이터가 존재하는 경로
        '''
        left_data_glob = glob.glob(path+'/data_left*.json')
        right_data_glob = glob.glob(path+'/data_right*.json')

        for data in left_data_glob:
            with open(data,'r') as f:
                fstr = f.read()
                f.close()
            jstr = json.loads(fstr)
            obj_points = jstr['objp']
            img_points = jstr['imgp']
            obj_points, img_points = np.array(obj_points, dtype=np.float32), np.array(img_points, dtype=np.float32)
            self.obj_datas.append(obj_points)
            self.left_datas.append(img_points)
        
        for data in right_data_glob:
            with open(data,'r') as f:
                fstr = f.read()
                f.close()
            jstr = json.loads(fstr)
            img_points = jstr['imgp']
            img_points = np.array(img_points, dtype=np.float32)
            self.right_datas.append(img_points)

    def load_ChArUco_datas(self, path='./ChAruco_datas'):
        '''
        # load_ChArUco_datas
        지정된 path로 부터 ChArUco 패턴 데이터를 불러오는 메서드

        ## Parameters
        - path

        ### path
        데이터가 존재하는 경로
        '''
        left_data_glob = glob.glob(path+'/data_left*.json')
        right_data_glob = glob.glob(path+'/data_right*.json')

        for data in left_data_glob:
            with open(data,'r') as f:
                fstr = f.read()
                f.close()
            jstr = json.loads(fstr)
            img_points = jstr['imgp']
            ids = jstr['ids']
            ids, img_points = np.array(ids, dtype=np.float32), np.array(img_points, dtype=np.float32)
            self.left_Ch_datas.append(img_points)
            self.left_Ch_ids.append(ids)
        for data in right_data_glob:
            with open(data,'r') as f:
                fstr = f.read()
                f.close()
            jstr = json.loads(fstr)
            img_points = jstr['imgp']
            ids = jstr['ids']
            ids, img_points = np.array(ids, dtype=np.float32), np.array(img_points, dtype=np.float32)
            self.right_Ch_datas.append(img_points)
            self.right_Ch_ids.append(ids)

    def run_calibration(self):
        '''
        # run_calibration
        스테레오 캘리브레이션을 수행하고 결과를 필드에 저장하는 메서드

        ## Parameters
        None
        '''
        _, mtx1, dist1, rvecs1, tvecs1 =cv.calibrateCamera(self.obj_datas, self.left_datas, self.img_size, None, None)
        _, mtx2, dist2, rvecs2, tvecs2 =cv.calibrateCamera(self.obj_datas, self.right_datas, self.img_size, None, None)
        
        ret, self.K1, self.D1,  \
        self.K2, self.D2,       \
        self.R, self.T,         \
        self.E, self.F= cv.stereoCalibrate(
             self.obj_datas, self.left_datas, self.right_datas,
              mtx1, dist1, mtx2, dist2, self.img_size,
              flags=cv.CALIB_FIX_INTRINSIC)
        self.is_calibrated = True
        mean_error = 0
        for i in range(len(self.obj_datas)):
            img_points_left, _ = cv.projectPoints(self.obj_datas[i], rvecs1[i], tvecs1[i], mtx1, dist1)
            error = cv.norm(self.left_datas[i], img_points_left, cv.NORM_L2) / len(img_points_left)
            mean_error += error
        print( "total error for left: {}".format(mean_error / len(self.obj_datas)) )
        mean_error = 0
        for i in range(len(self.obj_datas)):
            img_points_right, _ = cv.projectPoints(self.obj_datas[i], rvecs2[i], tvecs2[i], mtx2, dist2)
            error = cv.norm(self.right_datas[i], img_points_right, cv.NORM_L2) / len(img_points_right)
            mean_error += error
        print( "total error for right: {}".format(mean_error / len(self.obj_datas)) )
    
    # def run_calibration_with_ChArUco(self):
    #     '''
    #     스테레오 캘리브레이션을 수행하고 결과를 필드에 저장하는 메서드
    #     두 카메라의 패러미터를 ChArUco패턴을 통해 계산하는 것을 제외하면
    #     run_calibration 메서드와 동일
    #     '''
    #     _, mtx1, dist1, rvecs1, tvecs1, _, _ = cv.aruco.calibrateCameraCharuco(
    #                   charucoCorners=self.left_Ch_datas,
    #                   charucoIds=self.left_Ch_ids,
    #                   board=self.Charuco_board,
    #                   imageSize=self.img_size,
    #                   cameraMatrix=None,
    #                   distCoeffs=None,
    #                   flags=cv.CALIB_USE_INTRINSIC_GUESS,
    #                   criteria=(cv.TERM_CRITERIA_EPS & cv.TERM_CRITERIA_COUNT, 10000, 1e-9))
    #     _, mtx2, dist2, rvecs2, tvecs2, _, _ = cv.aruco.calibrateCameraCharuco(
    #                   charucoCorners=self.right_Ch_datas,
    #                   charucoIds=self.right_Ch_ids,
    #                   board=self.Charuco_board,
    #                   imageSize=self.img_size,
    #                   cameraMatrix=None,
    #                   distCoeffs=None,
    #                   flags=cv.CALIB_USE_INTRINSIC_GUESS,
    #                   criteria=(cv.TERM_CRITERIA_EPS & cv.TERM_CRITERIA_COUNT, 10000, 1e-9))
    #     ret, self.K1, self.D1,  \
    #     self.K2, self.D2,       \
    #     self.R, self.T,         \
    #     self.E, self.F= cv.stereoCalibrate(
    #          self.obj_datas, self.left_datas, self.right_datas,
    #           mtx1, dist1, mtx2, dist2, self.img_size,
    #           flags=cv.CALIB_FIX_INTRINSIC)
    #     self.is_calibrated = True

    def save_calibration_datas(self, directory='./calib'):
        '''
        # save_calibration_datas
        얻어낸 스테레오 카메라 패러미터를 path에 JSON형식으로 저장하는 메서드
        
        ## Parameters
        - directory

        ### directory
        캘리브레이션 데이터가 저장될 경로
        '''
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
                "imgSize" : self.img_size[::-1] 
            }
        )
        with open(directory+'/'+filename, 'w') as f:
            f.write(jsonfile_calib)
    
    def load_calibration_datas(self, filename='./calib/calib_data.json'):
        '''
        # load_calibration_datas
        저장된 JSON형식의 카메라 패러미터를 불러오는 메서드

        ## Parameters
        - filename

        ### filename
        저장된 카메라 패러미터 파일의 이름
        '''
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
        self.is_calibrated = True

def main():
    '''
    파일 실행시 자동으로 (640,480) 크기의 패턴 사진들을 불러와 캘리브레이션 수행 후, 
    "./calib" 폴더에 저장.
    '''
    calibrator = Calibrator((640,480))
    calibrator.load_datas('./datas')
    calibrator.run_calibration()
    calibrator.save_calibration_datas('./calib')

if __name__ == "__main__":
    main()