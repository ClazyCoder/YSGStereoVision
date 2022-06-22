# YSGStereoVision

스테레오 카메라로 부터 PointCloud를 얻어내어 [PointNet][pointnet]에
학습시켜 Semantic Segmentation을 수행하는 프로그램.

## 설치 (Installation)

    pip install -r requirements.txt

## 파일들

- cameramodule/cameramodule.py
- stereomodule/calibration.py
- stereomodule/stereomatcher.py
- captureChArUco.py
- captureChessBoardPatterns.py
- main.py
- PointNet_with_Custom_Dataset.ipynb

## 파일 설명

### cameramodule.py

두대의 카메라를 하나의 클래스로 묶어 관리하기위한 StereoCamera 클래스가 정의된 코드.

### calibration.py

스테레오 카메라로 부터 얻은 데이터를 통해 Calibration을 수행하고 카메라 패러미터를 저장하는 Calibrator 클래스가 정의된 코드.

### stereomatcher.py

Stereo Rectfication이 적용된 이미지들로 부터 Block Matching을 수행하고 Disparity map을 얻어내는 StereoMatcher 클래스가 정의된 코드.

### captureChArUco.py

StereoCamera 클래스를 이용하여 ChArUco 보드를 캡쳐하여 Calibration에 사용될 특징점들을 얻어내는 코드.

### captureChessBoardPatterns.py

StereoCamera 클래스를 이용하여 체커보드 패턴을 캡쳐하여 Calibration에 사용될 특징점들을 얻어내는 코드.

### main.py

실시간으로 Disparity를 계산하여 보여주는 코드.

키보드에서 'c'를 누를시 .ply의 형태로 PointCloud를 저장함.

### PointNet_with_Custom_Dataset.ipynb

main.py에서 얻어낸 PointCloud를 기반으로 학습을 수행하는 코드.

[pointnet]: https://openaccess.thecvf.com/content_cvpr_2017/html/Qi_PointNet_Deep_Learning_CVPR_2017_paper.html "CVPR 2017 PointNet"
