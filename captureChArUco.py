import cv2 as cv
import numpy as np
from cv2 import aruco
import CameraModule.CameraModule as CM
import os

def main():
    if not os.path.isdir('./Aruco_captures'):
        os.mkdir('./Aruco_captures')
    if not os.path.isdir('./Aruco_datas'):
        os.mkdir('./Aruco_datas')
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)
    imboard = board.draw((2000, 2000))
    cv.imwrite("./chessboard.tiff", imboard)

if __name__ == "__main__":
    main()