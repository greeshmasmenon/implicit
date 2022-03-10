from .matrix cimport CSRMatrix, Matrix


cdef extern from "implicit/gpu/als.h" namespace "implicit::gpu" nogil:
    cdef cppclass LeastSquaresSolver[T]:
        LeastSquaresSolver() except +

        void calculate_yty(const Matrix[T] & Y, Matrix[T] * YtY, float regularization) except *

        void least_squares(const CSRMatrix & Cui, Matrix[T] * X,
                           const Matrix[T] & YtY, const Matrix[T] & Y,
                           int cg_steps) except +

        float calculate_loss(const CSRMatrix & Cui, const Matrix[T] & X,
                             const Matrix[T] & Y, float regularization) except +
