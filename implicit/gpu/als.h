#ifndef IMPLICIT_GPU_ALS_H_
#define IMPLICIT_GPU_ALS_H_
#include "implicit/gpu/matrix.h"

// Forward ref: don't require the whole cublas definition here
struct cublasContext;

namespace implicit {
namespace gpu {

struct LeastSquaresSolver {
  explicit LeastSquaresSolver();
  ~LeastSquaresSolver();

  void least_squares(const CSRMatrix &Cui, Matrix<float> *X, const Matrix<float> &YtY,
                     const Matrix<float> &Y, int cg_steps) const;

  void calculate_yty(const Matrix<float> &Y, Matrix<float> *YtY, float regularization);

  float calculate_loss(const CSRMatrix &Cui, const Matrix<float> &X, const Matrix<float> &Y,
                       float regularization);

  cublasContext *blas_handle;
};
} // namespace gpu
} // namespace implicit
#endif // IMPLICIT_GPU_ALS_H_
