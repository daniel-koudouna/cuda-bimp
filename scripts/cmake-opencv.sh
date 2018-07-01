INSTALL_DIR="/usr/local"
CUDA_DIR="/usr/local/cuda-8.0"

cd "opencv-3.3.1/" && cd "build" &&
    cmake \
	-D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=$INSTALL_DIR \
	-D WITH_CUDA=ON \
	-D WITH_TBB=ON \
	-D WITH_V4L=ON \
	-D WITH_QT=ON \
	-D WITH_OPENGL=ON \
	-D CUDA_TOOLKIT_ROOT_DIR=$CUDA_DIR \
	-D OPENCV_EXTRA_MODULES_PATH="../../opencv_contrib-3.3.1/modules" \
	..

