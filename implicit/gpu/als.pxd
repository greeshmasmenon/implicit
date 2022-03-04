from .matrix cimport CSRMatrix, Matrix


cdef extern from "implicit/gpu/als.h" namespace "implicit::gpu" nogil:
    cdef cppclass LeastSquaresSolver:
        LeastSquaresSolver() except +

        void calculate_yty(const Matrix[float] & Y, Matrix[float] * YtY, float regularization) except *

        void least_squares(const CSRMatrix & Cui, Matrix[float] * X,
                           const Matrix[float] & YtY, const Matrix[float] & Y,
                           int cg_steps) except +

        float calculate_loss(const CSRMatrix & Cui, const Matrix[float] & X,
                             const Matrix[float] & Y, float regularization) except +
