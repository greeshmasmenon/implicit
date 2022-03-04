from .matrix cimport Matrix


cdef extern from "implicit/gpu/random.h" namespace "implicit::gpu" nogil:
    cdef cppclass RandomState:
        RandomState(long rows) except +
        Matrix[float] uniform(int rows, int cols, float low, float high) except +
        Matrix[float] randn(int rows, int cols, float mean, float stdev) except +
