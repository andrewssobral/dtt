# Data Transfer Tools for C++ Linear Algebra Libraries.
DTT is a **header-only** library that provides data transfer tools between C++ linear algebra libraries.
Currently, it supports data transfer between the following libraries:
* Eigen (>=3.3.x)
* Armadillo (>=9.x)
* OpenCV (cv::Mat) (2.x, 3.x, 4.x) 
* ArrayFire (>=3.x)
* LibTorch (PyTorch C++) (>=1.x)

Current status:

Last page update: 25/08/2019

| From/To   | Eigen | Armadillo | OpenCV | ArrayFire | LibTorch |
|-----------|:-----:|:---------:|:------:|:---------:|:--------:|
| Eigen     |   -   |     X     |    X   |     X     |     X    |
| Armadillo |   X   |     -     |    X   |     X     |     X    |
| OpenCV    |   X   |     X     |    -   |     X     |     X    |
| ArrayFire |   X   |     X     |    X   |     -     |     X    |
| LibTorch  |   X   |     X     |    X   |     X     |     -    |


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

How to use:
```c++
#include <dtt.h>
using namespace dtt;
// that's all!
```

List of available functions:

* From Eigen:
* * eigen2arma
* * eigen2cv
* * eigen2af
* * eigen2libtorch

* From Armadillo:
* * arma2eigen
* * arma2cv
* * arma2af
* * arma2libtorch

* From OpenCV:
* * cv2eigen
* * cv2arma
* * cv2af
* * cv2libtorch

* From ArrayFire:
* * af2eigen
* * af2arma
* * af2cv
* * af2libtorch

* From LibTorch:
* * libtorch2eigen
* * libtorch2arma
* * libtorch2cv
* * libtorch2af

See **test/dtt_test.h** and **test/dtt_test.cpp** for usage examples.
