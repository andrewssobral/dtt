/**
 * DTT - Data Transfer Tools for C++ Linear Algebra Libraries.
 * It supports data transfer between the following libraries:
 * Eigen, Armadillo, OpenCV, ArrayFire, LibTorch
 */
#include "dtt_test.h"

void debug_info() {
  std::cout << "Eigen version: " << EIGEN_WORLD_VERSION << EIGEN_MAJOR_VERSION << EIGEN_MINOR_VERSION << std::endl;
  std::cout << "Armadillo version: " << arma::arma_version::as_string() << std::endl;
  std::cout << "Using OpenCV version " << CV_VERSION << std::endl;
}

int main(int argc, const char **argv) {
  debug_info();
  test_libs();
  test_libs_conversion();
}
