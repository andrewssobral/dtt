# Data Transfer Tools for C++ Linear Algebra Libraries.
Currently, it supports data transfer between the following libraries:
* Eigen (>=3.3.x)
* Armadillo (>=9.x)
* OpenCV (cv::Mat) (2.x, 3.x, 4.x) 
* ArrayFire (>=3.x)
* LibTorch (PyTorch C++) (>=1.x)

Status:

| From/To   | Eigen | Armadillo | OpenCV | ArrayFire | LibTorch |
|-----------|:-----:|:---------:|:------:|:---------:|:--------:|
| Eigen     |   -   |     X     |    X   |     X     |     X    |
| Armadillo |   X   |     -     |    X   |    TODO   |   TODO   |
| OpenCV    |   X   |     X     |    -   |    TODO   |   TODO   |
| ArrayFire |   X   |    TODO   |  TODO  |     -     |   TODO   |
| LibTorch  |  TODO |    TODO   |  TODO  |    TODO   |     -    |


Tested on:
* MacBook Pro (13-inch, 2017)
* Mac OS X Mojave (10.14.5), Clang 10 (clang-1000.11.45.5), Xcode 10.1 with the following libraries:
* * OpenCV 4.1.1 (stable, built from source)
* * Eigen 3.3.7 (stable)
* * Armadillo 9.600.5 (stable)
* * ArrayFire 3.6.4 (stable)
* * LibTorch (1.3.0.dev20190820)

Install dependencies:
```
brew install eigen
brew install armadillo
# download and install ArrayFire: https://arrayfire.com/download/
# download and install LibTorch: https://pytorch.org/get-started/locally/
# download adn install OpenCV: https://opencv.org/releases/
```

How to compile and run:
```
git clone https://github.com/andrewssobral/dtt.git
cd dtt && mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_HOME ..
./dtt_test
```
