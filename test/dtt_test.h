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

//---------------------------------------------------------------------------
// Eigen to Armadillo, OpenCV, ArrayFire, LibTorch
//---------------------------------------------------------------------------

void test_eigen_arma() {
  print_line();
  std::cout << "Testing Eigen to Armadillo (copy 3x3 matrix):" << std::endl;
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

// TODO: test_eigen_arma with no-copy

void test_eigen_opencv() {
  print_line();
  std::cout << "Testing Eigen to OpenCV (copy 3x3 matrix):" << std::endl;
  // Eigen
  //Eigen::MatrixXd E = Eigen::MatrixXd::Random(3,3);
  Eigen::MatrixXf E = Eigen::MatrixXf::Random(3,3);
  std::cout << "EigenMat:\n" << E << std::endl;
  // OpenCV
  cv::Mat C;
  cv::eigen2cv(E, C);
  std::cout << "cvMat:\n" << C << std::endl;
  // re-check after changes
  //C.at<double>(0,0) = 0;
  C.at<float>(0,0) = 0;
  std::cout << "cvMat:\n" << C << std::endl;
  std::cout << "EigenMat:\n" << E << std::endl;
}

// TODO: test_eigen_opencv with no-copy

void test_eigen_af() {
  print_line();
  std::cout << "Testing Eigen to ArrayFire (copy 3x3 matrix):" << std::endl;
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

// TODO: test_eigen_af with no-copy

void test_eigen_libtorch1() { // CM = Column-major storage
  print_line();
  std::cout << "Testing Eigen(CM) to LibTorch (copy 3x3 matrix):" << std::endl;
  // Eigen
  //MatrixX<V> = Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic>
  //MatrixX<float> = MatrixXf
  Eigen::MatrixXf E(3,3);
//  E << 1.010101, 2.020202, 3.030303,
//       4.040404, 5.050505, 6.060606,
//       7.070707, 8.080808, 9.090909;
  E = Eigen::MatrixXf::Random(3,3);
  std::cout << "EigenMat:\n" << E << std::endl;
  // LibTorch
  torch::Tensor T = eigen2libtorch<float>(E);
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  // re-check after changes
  T[0][0] = 0;
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  std::cout << "EigenMat:\n" << E << std::endl;
}

void test_eigen_libtorch2() { // RM = Row-major storage (same as LibTorch)
  print_line();
  std::cout << "Testing Eigen(RM) to LibTorch (copy 3x3 matrix):" << std::endl;
  // Eigen
  // MatrixXrm<V> = Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  MatrixXrm<float> E(3,3);
  E << 1.010101, 2.020202, 3.030303,
  4.040404, 5.050505, 6.060606,
  7.070707, 8.080808, 9.090909;
  std::cout << "EigenMat:\n" << E << std::endl;
  // LibTorch
  torch::Tensor T = eigen2libtorch(E);
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  // re-check after changes
  T[0][0] = 0;
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  std::cout << "EigenMat:\n" << E << std::endl;
}

void test_eigen_libtorch3() { // RM = Row-major storage (same as LibTorch)
  print_line();
  std::cout << "Testing Eigen(RM) to LibTorch (no-copy 3x3 matrix):" << std::endl;
  // Eigen
  // MatrixXrm<V> = Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  MatrixXrm<float> E(3,3);
  E << 1.010101, 2.020202, 3.030303,
  4.040404, 5.050505, 6.060606,
  7.070707, 8.080808, 9.090909;
  std::cout << "EigenMat:\n" << E << std::endl;
  // LibTorch
  torch::Tensor T = eigen2libtorch(E, false);
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  // re-check after changes
  T[0][0] = 0;
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  std::cout << "EigenMat:\n" << E << std::endl;
}

//---------------------------------------------------------------------------
// Armadillo to Eigen, OpenCV, ArrayFire, LibTorch
//---------------------------------------------------------------------------

void test_arma_eigen() {
  print_line();
  std::cout << "Testing Armadillo to Eigen (copy 3x3 matrix):" << std::endl;
  // Armadillo
  arma::mat A = arma::randu<arma::mat>(3,3); // arma::mat = arma::Mat<double>
  std::cout << "ArmaMat:\n" << A << std::endl;
  // Eigen
  Eigen::MatrixXd E = arma2eigen(A);
  std::cout << "EigenMat:\n" << E << std::endl;
  // re-check after changes
  E(0,0) = 0;
  std::cout << "EigenMat:\n" << E << std::endl;
  std::cout << "ArmaMat:\n" << A << std::endl;
}

// TODO: test_arma_eigen with no-copy

void test_arma_opencv() {
  print_line();
  std::cout << "Testing Armadillo to OpenCV (copy 3x3 matrix):" << std::endl;
  // Armadillo
  //arma::mat A = arma::randu<arma::mat>(3,3); // arma::mat = arma::Mat<double>
  arma::Mat<float> A = arma::randu<arma::Mat<float>>(3,3);
  std::cout << "ArmaMat:\n" << A << std::endl;
  // OpenCV
  cv::Mat_<float> C(3,3);
  arma2cv<float>(A, C);
  std::cout << "cvMat:\n" << C << std::endl;
  // re-check after changes
  C.at<float>(0,0) = 0;
  std::cout << "cvMat:\n" << C << std::endl;
  std::cout << "ArmaMat:\n" << A << std::endl;
}

// TODO: test_arma_opencv with no-copy

void test_arma_af() {
  print_line();
  std::cout << "Testing Armadillo to ArrayFire (copy 3x3 matrix):" << std::endl;
  // Armadillo
  //arma::mat M = arma::randu<arma::mat>(3,3); // arma::mat = arma::Mat<double>
  arma::Mat<float> M = arma::randu<arma::Mat<float>>(3,3);
  std::cout << "ArmaMat:\n" << M << std::endl;
  // ArrayFire
  auto A = arma2af<float>(M);
  std::cout << "AfMat:" << std::endl;
  af_print(A);
  // re-check after changes
  A(1,1) = 0;
  std::cout << "AfMat:" << std::endl;
  af_print(A);
  std::cout << "ArmaMat:\n" << M << std::endl;
}

void test_arma_libtorch() {
  print_line();
  std::cout << "Testing Armadillo to LibTorch (copy 3x3 matrix):" << std::endl;
  // Armadillo
  //arma::mat A = arma::randu<arma::mat>(3,3); // arma::mat = arma::Mat<double>
  arma::Mat<float> A = arma::randu<arma::Mat<float>>(3,3);
  std::cout << "ArmaMat:\n" << A << std::endl;
  // LibTorch
  torch::Tensor T = arma2libtorch<float>(A);
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  // re-check after changes
  T[0][0] = 0;
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  std::cout << "ArmaMat:\n" << A << std::endl;
}

//---------------------------------------------------------------------------
// OpenCV to Eigen, Armadillo, ArrayFire, LibTorch
//---------------------------------------------------------------------------

void test_opencv_eigen() {
  print_line();
  std::cout << "Testing OpenCV to Eigen (copy 3x3 matrix):" << std::endl;
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

// TODO: test_opencv_eigen with no-copy

void test_opencv_arma() {
  print_line();
  std::cout << "Testing OpenCV to Armadillo (copy 3x3 matrix):" << std::endl;
  // OpenCV
  cv::Mat C(3, 3, CV_32FC1);
  cv::randn(C, 0.0f, 1.0f);
  std::cout << "cvMat:\n" << C << std::endl;
  // Armadillo
  auto A = cv2arma<float>(C);
  arma::inplace_trans(A);
  std::cout << "ArmaMat:\n" << A << std::endl;
  // re-check after changes
  A(0,0) = 0;
  std::cout << "ArmaMat:\n" << A << std::endl;
  std::cout << "cvMat:\n" << C << std::endl;
}

// TODO: test_opencv_arma with no-copy

void test_opencv_af() {
  print_line();
  std::cout << "Testing OpenCV to ArrayFire (copy 3x3 matrix):" << std::endl;
  // OpenCV
  cv::Mat C(3, 3, CV_32FC1);
  cv::randn(C, -1.0f, 1.0f);
  std::cout << "cvMat:\n" << C << std::endl;
  // ArrayFire
  auto A = cv2af<float>(C);
  af::transposeInPlace(A);
  std::cout << "AfMat:" << std::endl;
  af_print(A);
  // re-check after changes
  A(0,0) = 0;
  std::cout << "AfMat:" << std::endl;
  af_print(A);
  std::cout << "cvMat:\n" << C << std::endl;
}

void test_opencv_libtorch1() {
  print_line();
  std::cout << "Testing OpenCV to LibTorch (copy 3x3 matrix):" << std::endl;
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
  std::cout << "Testing OpenCV to LibTorch (no-copy 3x3 matrix):" << std::endl;
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

//---------------------------------------------------------------------------
// ArrayFire to Eigen, Armadillo, OpenCV, LibTorch
//---------------------------------------------------------------------------

void test_af_eigen() {
  print_line();
  //af::info();
  std::cout << "Testing ArrayFire to Eigen (copy 3x3 matrix):" << std::endl;
  // ArrayFire
  af::array A = af::randu(3,3, f32);
  std::cout << "AfMat:" << std::endl;
  af_print(A);
  //float* data = A.host<float>();
  //Eigen::Map<Eigen::MatrixXf> E(data, A.dims(0), A.dims(1));
  //Eigen::MatrixXf E = af2eigen(A);
  auto E = af2eigen<float>(A);
  std::cout << "EigenMat:\n" << E << std::endl;
  // re-check after changes
  E(0,0) = 0;
  std::cout << "EigenMat:\n" << E << std::endl;
  std::cout << "AfMat:" << std::endl;
  af_print(A);
}

// TODO: test_af_eigen with no-copy (same device)

void test_af_arma() {
  print_line();
  //af::info();
  std::cout << "Testing ArrayFire to Armadillo (copy 3x3 matrix):" << std::endl;
  // ArrayFire
  af::array A = af::randu(3,3, f32);
  std::cout << "AfMat:" << std::endl;
  af_print(A);
  // Armadillo
  auto M = af2arma<float>(A);
  //arma::inplace_trans(A);
  std::cout << "ArmaMat:\n" << M << std::endl;
  // re-check after changes
  M(0,0) = 0;
  std::cout << "ArmaMat:\n" << M << std::endl;
  std::cout << "AfMat:" << std::endl;
  af_print(A);
}

// TODO: test_af_arma with no-copy (same device)

void test_af_cv() {
  print_line();
  //af::info();
  std::cout << "Testing ArrayFire to OpenCV (copy 3x3 matrix):" << std::endl;
  // ArrayFire
  af::array A = af::randu(3,3,f32); // first row, second column.
  std::cout << "AfMat:" << std::endl;
  af_print(A);
  // OpenCV
  auto C = af2cv<float>(A);
  std::cout << "cvMat:\n" << C << std::endl;
  // re-check after changes
  C.at<float>(0,0) = 0;
  std::cout << "cvMat:\n" << C << std::endl;
  std::cout << "AfMat:" << std::endl;
  af_print(A);
}

void test_af_libtorch() {
  print_line();
  //af::info();
  std::cout << "Testing ArrayFire to LibTorch (copy 3x3 matrix):" << std::endl;
  // ArrayFire
  af::array A = af::randu(3,3,f32); // first row, second column.
  std::cout << "AfMat:" << std::endl;
  af_print(A);
  // LibTorch
  torch::Tensor T = af2libtorch<float>(A);
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  // re-check after changes
  T[0][0] = 0;
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  std::cout << "AfMat:" << std::endl;
  af_print(A);
}

//---------------------------------------------------------------------------
// LibTorch to Eigen, Armadillo, OpenCV, ArrayFire
//---------------------------------------------------------------------------

void test_libtorch_eigen1() {
  print_line();
  std::cout << "Testing LibTorch to Eigen (copy 3x3 matrix):" << std::endl;
  // LibTorch
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  torch::Tensor T = torch::rand({3, 3});
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  // Eigen
  auto E = libtorch2eigen<float>(T);
  std::cout << "EigenMat:\n" << E << std::endl;
  // re-check after changes
  E(0,0) = 0;
  std::cout << "EigenMat:\n" << E << std::endl;
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
}

void test_libtorch_eigen2() {
  print_line();
  std::cout << "Testing LibTorch to Eigen (no-copy 3x3 matrix):" << std::endl;
  // LibTorch
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  torch::Tensor T = torch::rand({3, 3});
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  // Eigen
  Eigen::Map<MatrixXrm<float>> E(T.data_ptr<float>(), T.size(0), T.size(1));
  std::cout << "EigenMat:\n" << E << std::endl;
  // re-check after changes
  E(0,0) = 0;
  std::cout << "EigenMat:\n" << E << std::endl;
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
}

void test_libtorch_arma1() {
  print_line();
  std::cout << "Testing LibTorch to Armadillo (copy 3x3 matrix):" << std::endl;
  // LibTorch
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  torch::Tensor T = torch::rand({3, 3});
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  // Armadillo
  auto A = libtorch2arma<float>(T);
  arma::inplace_trans(A);
  std::cout << "ArmaMat:\n" << A << std::endl;
  // re-check after changes
  A(0,0) = 0;
  std::cout << "ArmaMat:\n" << A << std::endl;
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
}

void test_libtorch_arma2() {
  print_line();
  std::cout << "Testing LibTorch to Armadillo (no-copy 3x3 matrix):" << std::endl;
  // LibTorch
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  torch::Tensor T = torch::rand({3, 3});
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  // Armadillo
  auto A = libtorch2arma<float>(T, /*copy*/false);
  std::cout << "ArmaMat:\n" << A << std::endl;
  std::cout << ">>> Note that ArmaMat is transposed <<<" << std::endl;
  // re-check after changes
  A(0,0) = 0;
  std::cout << "ArmaMat:\n" << A << std::endl;
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
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

void test_libtorch_af() {
  print_line();
  std::cout << "Testing LibTorch to ArrayFire (copy 3x3 matrix):" << std::endl;
  // LibTorch
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  torch::Tensor T = torch::rand({3, 3});
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
  // ArrayFire
  auto A = libtorch2af<float>(T);
  af::transposeInPlace(A);
  std::cout << "AfMat:" << std::endl;
  af_print(A);
  // re-check after changes
  A(0,0) = 0;
  std::cout << "AfMat:" << std::endl;
  af_print(A);
  std::cout << "LibTorch:" << std::endl;
  std::cout << T << std::endl;
}

// TODO: test_libtorch_af with no-copy (if same device)

//---------------------------------------------------------------------------
// Call functions
//---------------------------------------------------------------------------

void test_libs_conversion() {
  test_eigen_opencv();
  test_eigen_arma();
  test_eigen_af();
  test_eigen_libtorch1();
  test_eigen_libtorch2();
  test_eigen_libtorch3();
  test_arma_eigen();
  test_arma_opencv();
  test_arma_af();
  test_arma_libtorch();
  test_opencv_eigen();
  test_opencv_arma();
  test_opencv_af();
  test_opencv_libtorch1();
  test_opencv_libtorch2();
  test_af_eigen();
  test_af_arma();
  test_af_cv();
  test_af_libtorch();
  test_libtorch_eigen1();
  test_libtorch_eigen2();
  test_libtorch_arma1();
  test_libtorch_arma2();
  test_libtorch_opencv1();
  test_libtorch_opencv2();
  test_libtorch_af();
}

//---------------------------------------------------------------------------
// Set of functions to test each library
//---------------------------------------------------------------------------

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
