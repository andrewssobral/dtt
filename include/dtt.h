/**
 * DTT - Data Transfer Tools for C++ Linear Algebra Libraries.
 * It supports data transfer between the following libraries:
 * Eigen, Armadillo, OpenCV, ArrayFire, LibTorch
 */
#pragma once

#include <iostream>
#include <cstdio>
#include <cstdlib>

#include <Eigen/Dense>
#include <armadillo>
#include <arrayfire.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#if CV_MAJOR_VERSION >= 4
#define CV_BGR2RGB cv::COLOR_BGR2RGB
#endif

namespace dtt {

  // same as MatrixXf, but with row-major memory layout
  //typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;
  
  // MatrixXrm<float> x; instead of MatrixXf_rm x;
  template <typename T>
  using MatrixXrm = typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  
  // MatrixX<float> x; instead of Eigen::MatrixXf x;
  template <typename T>
  using MatrixX = typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  
  //---------------------------------------------------------------------------
  // Eigen to Armadillo, OpenCV, ArrayFire, LibTorch, File
  //---------------------------------------------------------------------------
  
  arma::mat eigen2arma(Eigen::MatrixXd& E, bool copy=true) {
    return arma::mat(E.data(), E.rows(), E.cols(), /*copy_aux_mem*/copy, /*strict*/false);
  }
  
  af::array eigen2af(Eigen::MatrixXf& E){
    af::array A(E.rows(), E.cols(), E.data());
    return A;
  }
  
  //void eigen2cv(const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& src, cv::Mat& dst)
  
  template <typename V>
  torch::Tensor eigen2libtorch(MatrixX<V> &M) {
    Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> E(M);
    std::vector<int64_t> dims = {E.rows(), E.cols()};
    auto T = torch::from_blob(E.data(), dims).clone(); //.to(torch::kCPU);
    return T;
  }
  
  template <typename V>
  torch::Tensor eigen2libtorch(MatrixXrm<V> &E, bool copydata=true) {
    std::vector<int64_t> dims = {E.rows(), E.cols()};
    auto T = torch::from_blob(E.data(), dims);
    if (copydata)
      return T.clone();
    else
      return T;
  }
  
  void eigen2file(Eigen::MatrixXf E, std::string filename) {
    std::ofstream file(filename);
    if (file.is_open())
      file << E << '\n';
    file.close();
  }
  
  //---------------------------------------------------------------------------
  // Armadillo to Eigen, OpenCV, ArrayFire, File
  //---------------------------------------------------------------------------
  
  Eigen::MatrixXd arma2eigen(arma::mat& A) {
    return Eigen::Map<Eigen::MatrixXd>(A.memptr(), A.n_rows, A.n_cols);
  }
  
  template<class T>
  void arma2cv(const arma::Mat<T>& A, cv::Mat_<T>& C) {
    cv::transpose(cv::Mat_<T>(static_cast<int>(A.n_cols),
                              static_cast<int>(A.n_rows),
                              const_cast<T*>(A.memptr())), C);
  };
  
  /*
   For arma2af<float>, consider arma::Mat as a float matrix.
   Returns a copy of arma::Mat data.
   
   The root matrix class is arma::Mat<type>, where type is one of:
   float, double, std::complex<float>, std::complex<double>, short, int, long, and unsigned versions of short, int, long
   arma::mat  = arma::Mat<double>
   arma::dmat = arma::Mat<double>
   arma::fmat = arma::Mat<float>
   
   arma::mat  A = arma::randu<arma::mat>(5,5);
   arma::fmat B = arma::conv_to<arma::fmat>::from(A);
   */
  //reinterpret_cast
  //af::array arma2af(arma::mat& M){
  template<typename T>
  af::array arma2af(arma::Mat<T>& M) {
    af::array A(static_cast<int>(M.n_cols), static_cast<int>(M.n_rows), const_cast<T*>(M.memptr()));
    //af::transposeInPlace(A);
    return A;
  }
  
  void arma2file(arma::mat& A, std::string filename) {
    A.save(filename);
  }

  //---------------------------------------------------------------------------
  // OpenCV to Eigen, Armadillo, ArrayFire, LibTorch, File
  //---------------------------------------------------------------------------
  
  //void cv2eigen(const Mat& src, Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& dst)
  
  template<typename T>
  MatrixX<T> cv2eigen(cv::Mat &C) {
    Eigen::Map<MatrixXrm<T>> E(C.ptr<T>(), C.rows, C.cols);
    return E;
  }
  
  template<class T>
  arma::Mat<T> cv2arma(cv::Mat &C, bool copy=true) {
    /*
     OpenCV (cv::Mat) is Row-major order and Armadillo is Column-major order.
     If copy=true, arma::inplace_trans(A); should be used to keep
     the Row-major order from cv::Mat.
     */
    //return arma::Mat<T>(cvMatIn.data, cvMatIn.rows, cvMatIn.cols, false, false);
    return arma::Mat<T>(reinterpret_cast<T*>(C.data),
                        static_cast<arma::uword>(C.cols),
                        static_cast<arma::uword>(C.rows),
                        /*copy_aux_mem*/copy,
                        /*strict*/false);
  }
  
  /*
   For cv2af<float>, consider cv::Mat as a float matrix.
   Should be used with af::transposeInPlace(A);
   to keep the Row-major order from OpenCV.
   Returns a copy of cv::Mat data.
   */
  template<typename T>
  af::array cv2af(cv::Mat &C){
    af::array A(C.cols, C.rows, reinterpret_cast<T*>(C.data));
    return A;
  }
  
  torch::Tensor cv2libtorch(cv::Mat &C, bool copydata=true, bool is_cv_image=false) {
    int kCHANNELS = C.channels();
    if (is_cv_image) {
      if (kCHANNELS == 3) {
        cv::cvtColor(C, C, CV_BGR2RGB);
        C.convertTo(C, CV_32FC3, 1.0f / 255.0f);
      } else // considering channels = 1
        C.convertTo(C, CV_32FC1, 1.0f / 255.0f);
      auto T = torch::from_blob(C.data, {1, C.rows, C.cols, kCHANNELS});
      T = T.permute({0, 3, 1, 2});
      return T;
    } else {
      std::vector<int64_t> dims;
      at::TensorOptions options(at::kFloat);
      if (kCHANNELS == 1) {
        C.convertTo(C, CV_32FC1);
        dims = {C.rows, C.cols};
      }
      else { // considering channels = 3
        C.convertTo(C, CV_32FC3);
        dims = {C.rows, C.cols, kCHANNELS};
      }
      //auto T = torch::from_blob(C.ptr<float>(), dims, options).clone();
      //auto T = torch::from_blob(C.data, at::IntList(dims), options);
      auto T = torch::from_blob(C.data, dims, options);
      if (copydata)
        return T.clone();
      else
        return T;
    }
  }
  
  void cv2file(cv::Mat M, std::string filename) {
    std::ofstream file(filename);
    if (file.is_open())
      file << M << '\n';
    file.close();
  }
  
  //---------------------------------------------------------------------------
  // ArrayFire to Eigen, Armadillo
  //---------------------------------------------------------------------------
//  Eigen::MatrixXf af2eigen(af::array &A) {
//    float* data = A.host<float>();
//    Eigen::Map<Eigen::MatrixXf> E(data, A.dims(0), A.dims(1));
//    return E;
//  }
  
  template<typename T>
  MatrixX<T> af2eigen(af::array &A) {
    Eigen::Map<MatrixX<T>> E(A.host<T>(), A.dims(0), A.dims(1));
    return E;
  }
  
  template<typename T>
  arma::Mat<T> af2arma(af::array &A, bool copy=true) {
    return arma::Mat<T>(reinterpret_cast<T*>(A.host<T>()),
                        static_cast<arma::uword>(A.dims(0)),
                        static_cast<arma::uword>(A.dims(1)),
                        /*copy_aux_mem*/copy,
                        /*strict*/false);
  }
  
  //---------------------------------------------------------------------------
  // LibTorch to Eigen, Armadillo, OpenCV, ArrayFire
  //---------------------------------------------------------------------------
  
  template<typename V>
  Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic> libtorch2eigen(torch::Tensor &Tin) {
    /*
     LibTorch is Row-major order and Eigen is Column-major order.
     MatrixXrm uses Eigen::RowMajor for compatibility.
     */
    auto T = Tin.to(torch::kCPU);
    Eigen::Map<MatrixXrm<V>> E(T.data_ptr<V>(), T.size(0), T.size(1));
    return E;
  }
  
  template<typename V>
  arma::Mat<V> libtorch2arma(torch::Tensor &Tin, bool copy=true) {
    /*
     LibTorch is Row-major order and Armadillo is Column-major order.
     If copy=true, arma::inplace_trans(A); should be used to keep
     the Row-major order from LibTorch.
     */
    auto T = Tin.to(torch::kCPU);
    return arma::Mat<V>(reinterpret_cast<V*>(T.data_ptr<V>()),
                        static_cast<arma::uword>(T.size(0)),
                        static_cast<arma::uword>(T.size(1)),
                        /*copy_aux_mem*/copy,
                        /*strict*/false);
  }
  
  // Consider torch::Tensor as a float matrix
  cv::Mat libtorch2cv(torch::Tensor &Tin, bool copy=true) {
    auto T = Tin.to(torch::kCPU);
    cv::Mat C;
    if (copy) {
      cv::Mat M(T.size(0), T.size(1), CV_32FC1, T.data<float>());
      M.copyTo(C);
    } else
      C = cv::Mat(T.size(0), T.size(1), CV_32FC1, T.data<float>());
    return C;
  }
  
  /*
   If libtorch2af<float>, consider torch::Tensor as a float matrix.
   Should be used with af::transposeInPlace(A);
   to keep the Row-major order from LibTorch.
   Returns a copy of torch::Tensor data.
   */
  template<typename V>
  af::array libtorch2af(torch::Tensor &Tin){
    auto T = Tin.to(torch::kCPU);
    af::array A(T.size(0), T.size(1), T.data<V>());
    return A;
  }

}
