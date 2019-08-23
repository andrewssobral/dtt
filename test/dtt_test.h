/**
 * DTT - Data Transfer Tools for C++ Linear Algebra Libraries.
 * It supports data transfer between the following libraries:
 * Eigen, Armadillo, OpenCV, ArrayFire, LibTorch
 */
#pragma once

#include <dtt.h>

using namespace dtt;

void print_line() {
  std::cout << std::string(50, '-') << std::endl;
}

void test_opencv_eigen() {
  print_line();
  std::cout << "Testing OpenCV to Eigen:" << std::endl;
  // OpenCV
  cv::Mat C(3, 3, CV_32FC1);
  cv::randn(C, 0.0f, 1.0f);
  std::cout << "cvMat:\n" << C << std::endl;
  // Eigen
  Eigen::MatrixXd E;
  cv::cv2eigen(C, E);
  //auto E = cv2eigen<float>(C);
  std::cout << "EigenMat:\n" << E << std::endl;
  // re-check after changes
  E(0,0) = 0;
  std::cout << "EigenMat:\n" << E << std::endl;
  std::cout << "cvMat:\n" << C << std::endl;
}

void test_eigen_opencv() {
  print_line();
  std::cout << "Testing Eigen to OpenCV:" << std::endl;
  // Eigen
  Eigen::MatrixXd E = Eigen::MatrixXd::Random(3,3);
  std::cout << "EigenMat:\n" << E << std::endl;
  // OpenCV
  cv::Mat C;
  cv::eigen2cv(E, C);
  std::cout << "cvMat:\n" << C << std::endl;
  // re-check after changes
  C.at<double>(0,0) = 0;
  std::cout << "cvMat:\n" << C << std::endl;
  std::cout << "EigenMat:\n" << E << std::endl;
}

void test_opencv_arma() {
  print_line();
  std::cout << "Testing OpenCV to Armadillo:" << std::endl;
  // OpenCV
  cv::Mat C(3, 3, CV_32FC1);
  cv::randn(C, 0.0f, 1.0f);
  std::cout << "cvMat:\n" << C << std::endl;
  // Armadillo
  auto A = cv2arma<float>(C);
  std::cout << "ArmaMat:\n" << A << std::endl;
  // re-check after changes
  A(0,0) = 0;
  std::cout << "ArmaMat:\n" << A << std::endl;
  std::cout << "cvMat:\n" << C << std::endl;
}

void test_arma_opencv() {
  print_line();
  std::cout << "Testing Armadillo to OpenCV:" << std::endl;
  // Armadillo
  arma::mat A = arma::randu<arma::mat>(3,3);
  std::cout << "ArmaMat:\n" << A << std::endl;
  // OpenCV
  cv::Mat_<double> C(3,3);
  arma2cv<double>(A, C);
  std::cout << "cvMat:\n" << C << std::endl;
  // re-check after changes
  C.at<double>(0,0) = 0;
  std::cout << "cvMat:\n" << C << std::endl;
  std::cout << "ArmaMat:\n" << A << std::endl;
}

void test_eigen_arma() {
  print_line();
  std::cout << "Testing Eigen to Armadillo:" << std::endl;
  // Eigen
  Eigen::MatrixXd E = Eigen::MatrixXd::Random(3,3);
  std::cout << "EigenMat:\n" << E << std::endl;
  // Armadillo
  arma::mat A = eigen2arma(E);
  std::cout << "ArmaMat:\n" << A << std::endl;
  // re-check after changes
  A(0,0) = 0;
  std::cout << "ArmaMat:\n" << A << std::endl;
  std::cout << "EigenMat:\n" << E << std::endl;
}

void test_arma_eigen() {
  print_line();
  std::cout << "Testing Armadillo to Eigen:" << std::endl;
  // Armadillo
  arma::mat A = arma::randu<arma::mat>(3,3);
  std::cout << "ArmaMat:\n" << A << std::endl;
  // Eigen
  Eigen::MatrixXd E = arma2eigen(A);
  std::cout << "EigenMat:\n" << E << std::endl;
  // re-check after changes
  E(0,0) = 0;
  std::cout << "EigenMat:\n" << E << std::endl;
  std::cout << "ArmaMat:\n" << A << std::endl;
}

void test_eigen_af() {
  print_line();
  std::cout << "Testing Eigen to ArrayFire:" << std::endl;
  // Eigen
  Eigen::MatrixXf E = Eigen::MatrixXf::Random(3,3);
  std::cout << "EigenMat:\n" << E << std::endl;
  // ArrayFire
  //af::array A(E.rows(), E.cols(), E.data());
  af::array A = eigen2af(E);
  std::cout << "AfMat:" << std::endl;
  af_print(A);
  // re-check after changes
  A(0,0) = 0;
  std::cout << "AfMat:" << std::endl;
  af_print(A);
  std::cout << "EigenMat:\n" << E << std::endl;
}

void test_af_eigen() {
  print_line();
  std::cout << "Testing ArrayFire to Eigen:" << std::endl;
  // ArrayFire
  af::array A = af::randu(3,3, f32);
  std::cout << "AfMat:" << std::endl;
  af_print(A);
  //float* data = A.host<float>();
  //Eigen::Map<Eigen::MatrixXf> E(data, A.dims(0), A.dims(1));
  Eigen::MatrixXf E = af2eigen(A);
  std::cout << "EigenMat:\n" << E << std::endl;
  // re-check after changes
  E(0,0) = 0;
  std::cout << "EigenMat:\n" << E << std::endl;
  std::cout << "AfMat:" << std::endl;
  af_print(A);
}

void test_libtorch_eigen() {
  print_line();
  std::cout << "Testing LibTorch to Eigen:" << std::endl;
  // LibTorch
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  torch::Tensor T = torch::rand({3, 3});
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  // Eigen
  float* data = T.data_ptr<float>();
  Eigen::Map<Eigen::MatrixXf> E(data, T.size(0), T.size(1));
  std::cout << "EigenMat:\n" << E << std::endl;
  // re-check after changes
  E(0,0) = 0;
  std::cout << "EigenMat:\n" << E << std::endl;
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
}

void test_opencv_libtorch1() {
  print_line();
  std::cout << "Testing OpenCV to LibTorch #1 (copy 3x3 matrix):" << std::endl;
  // OpenCV
  cv::Mat C(3, 3, CV_32FC1);
  cv::randn(C, 0.0f, 1.0f);
  std::cout << "cvMat:\n" << C << std::endl;
  // LibTorch
  torch::Tensor T = cv2libtorch(C);
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  // re-check after changes
  T[0][0] = 0;
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  std::cout << "cvMat:\n" << C << std::endl;
}

void test_opencv_libtorch2() {
  print_line();
  std::cout << "Testing OpenCV to LibTorch #1 (no-copy 3x3 matrix):" << std::endl;
  // OpenCV
  cv::Mat C(3, 3, CV_32FC1);
  cv::randn(C, 0.0f, 1.0f);
  std::cout << "cvMat:\n" << C << std::endl;
  // LibTorch
  torch::Tensor T = cv2libtorch(C, false);
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  // re-check after changes
  T[0][0] = 0;
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  std::cout << "cvMat:\n" << C << std::endl;
}

void test_libtorch_opencv1() {
  print_line();
  std::cout << "Testing LibTorch to OpenCV (copy 3x3 matrix):" << std::endl;
  // LibTorch
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  torch::Tensor T = torch::rand({3, 3});
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  // OpenCV
  cv::Mat C = libtorch2cv(T);
  std::cout << "cvMat:\n" << C << std::endl;
  // re-check after changes
  C.at<float>(0,0) = 0;
  std::cout << "cvMat:\n" << C << std::endl;
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
}

void test_libtorch_opencv2() {
  print_line();
  std::cout << "Testing LibTorch to OpenCV (no-copy 3x3 matrix):" << std::endl;
  // LibTorch
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  torch::Tensor T = torch::rand({3, 3});
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  // OpenCV
  cv::Mat C = libtorch2cv(T, false);
  std::cout << "cvMat:\n" << C << std::endl;
  // re-check after changes
  C.at<float>(0,0) = 0;
  std::cout << "cvMat:\n" << C << std::endl;
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
}

void test_libs_conversion() {
  test_opencv_eigen();
  test_eigen_opencv();
  test_opencv_arma();
  test_arma_opencv();
  test_eigen_arma();
  test_arma_eigen();
  test_eigen_af();
  test_af_eigen();
  test_libtorch_eigen();
  test_opencv_libtorch1();
  test_opencv_libtorch2();
  test_libtorch_opencv1();
  test_libtorch_opencv2();
}

void test_opencv() {
  print_line();
  std::cout << "Testing OpenCV" << std::endl;
  cv::Mat C(3, 3, CV_32FC1);
  cv::randn(C, 0.0f, 1.0f);
  std::cout << C << std::endl;
}

void test_eigen() {
  print_line();
  std::cout << "Testing Eigen" << std::endl;
  Eigen::MatrixXd E(2,2);
  E(0,0) = 3;
  E(1,0) = 2.5;
  E(0,1) = -1;
  E(1,1) = E(1,0) + E(0,1);
  std::cout << E << std::endl;
}

void test_armadillo() {
  print_line();
  std::cout << "Testing Armadillo" << std::endl;
  arma::Mat<double> A = arma::randu(3,3);
  std::cout << "A:\n" << A << "\n";
}

void test_arrayfire() {
  print_line();
  std::cout << "Testing ArrayFire" << std::endl;
  try {
    //int device = argc > 1 ? atoi(argv[1]) : 0;
    //af::setDevice(device);
    af::info();
    af::array a = af::randu(100);
    double sum = af::sum<float>(a);
    printf("sum: %g\n", sum);
  } catch (af::exception& e) {
    fprintf(stderr, "%s\n", e.what());
    throw;
  }
}

void test_libtorch() {
  print_line();
  std::cout << "Testing LibTorch" << std::endl;
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  torch::Tensor T = torch::rand({2, 3});
  std::cout << T << std::endl;
}

void test_libs() {
  test_opencv();
  test_eigen();
  test_armadillo();
  test_arrayfire();
  test_libtorch();
}
