#ifndef IMPLICIT_GPU_ALS_H_
#define IMPLICIT_GPU_ALS_H_
#include "implicit/gpu/matrix.h"

// Forward ref: don't require the whole cublas definition here
struct cublasContext;

namespace implicit {
namespace gpu {

template<typename T>
struct LeastSquaresSolver {
  explicit LeastSquaresSolver();
  ~LeastSquaresSolver();

  void least_squares(const CSRMatrix &Cui, Matrix<T> *X, const Matrix<T> &YtY,
                     const Matrix<T> &Y, int cg_steps) const;

  void calculate_yty(const Matrix<T> &Y, Matrix<T> *YtY, float regularization);

  float calculate_loss(const CSRMatrix &Cui, const Matrix<T> &X, const Matrix<T> &Y,
                       float regularization);

  cublasContext *blas_handle;
};
} // namespace gpu
} // namespace implicit
#endif // IMPLICIT_GPU_ALS_H_
