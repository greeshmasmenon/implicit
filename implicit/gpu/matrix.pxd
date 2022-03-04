from libcpp cimport bool


cdef extern from "implicit/gpu/matrix.h" namespace "implicit::gpu" nogil:
    cdef cppclass CSRMatrix:
        CSRMatrix(int rows, int cols, int nonzeros,
                  const int * indptr, const int * indices, const float * data) except +

    cdef cppclass COOMatrix:
        COOMatrix(int rows, int cols, int nonzeros,
                  const int * row, const int * col, const float * data) except +

    cdef cppclass Vector[T]:
        Vector(int size, T * data) except +
        void to_host(T * output) except +
        T * data
        int size

    cdef cppclass Matrix[T]:
        Matrix(int rows, int cols, T * data, bool host) except +
        Matrix(const Matrix[T] & other, int rowid) except +
        Matrix(const Matrix[T] & other, int start, int end) except +
        Matrix(const Matrix[T] & other, const Vector[int] & rowids) except +
        Matrix(Matrix[T] && other) except +
        void to_host(T * output) except +
        void resize(int rows, int cols) except +
        void assign_rows(const Vector[int] & rowids, const Matrix[T] & other) except +
        Matrix[T] calculate_norms() except +
        int rows, cols
        T * data


    cdef cppclass KnnQuery:
        KnnQuery()
        void query(const Matrix & items, const Matrix & queries, int k,
                   int * indices, float * distances) except +
        void argpartition(const Matrix & items, int k, int * indices, float * distances) except +
        void argsort(Matrix * items, int * indices) except +