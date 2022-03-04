#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include "implicit/gpu/convert.cuh"
#include "implicit/gpu/dot.cuh"
#include "implicit/gpu/matrix.h"
#include "implicit/gpu/utils.h"

namespace implicit {
namespace gpu {
template <typename T>
Vector<T>::Vector(int size, const T *host_data)
    : size(size),
      storage(new rmm::device_uvector<T>(size, rmm::cuda_stream_view())),
      data(storage->data()) {
  if (host_data) {
    CHECK_CUDA(
        cudaMemcpy(data, host_data, size * sizeof(T), cudaMemcpyHostToDevice));
  }
}

template <typename T> void Vector<T>::to_host(T *out) const {
  CHECK_CUDA(cudaMemcpy(out, data, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template struct Vector<char>;
template struct Vector<int>;
template struct Vector<float>;

template <typename T>
Matrix<T>::Matrix(const Matrix<T> &other, int rowid)
    : rows(1), cols(other.cols), data(other.data + rowid * other.cols),
      storage(other.storage) {
  if (rowid >= other.rows) {
    throw std::invalid_argument("row index out of bounds for matrix");
  }
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T> &other, int start_rowid, int end_rowid)
    : rows(end_rowid - start_rowid), cols(other.cols),
      data(other.data + start_rowid * other.cols), storage(other.storage) {
  if (end_rowid < start_rowid) {
    throw std::invalid_argument("end_rowid < start_rowid for matrix slice");
  }
  if (end_rowid > other.rows) {
    throw std::invalid_argument("row index out of bounds for matrix");
  }
}

template <typename T>
void copy_rowids(const T *input, const int *rowids, int rows, int cols,
                 T *output) {
  // copy rows over
  auto count = thrust::make_counting_iterator<int>(0);
  thrust::for_each(count, count + (rows * cols), [=] __device__(int i) {
    int col = i % cols;
    int row = rowids[i / cols];
    output[i] = input[col + row * cols];
  });
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T> &other, const Vector<int> &rowids)
    : rows(rowids.size), cols(other.cols) {
  storage.reset(
      new rmm::device_uvector<T>(rows * cols, rmm::cuda_stream_view()));
  data = storage->data();
  copy_rowids(other.data, rowids.data, rows, cols, data);
}

template <typename T>
Matrix<T>::Matrix(int rows, int cols, T *host_data, bool allocate)
    : rows(rows), cols(cols) {
  if (allocate) {
    storage.reset(
        new rmm::device_uvector<T>(rows * cols, rmm::cuda_stream_view()));
    data = storage->data();
    if (host_data) {
      CHECK_CUDA(cudaMemcpy(data, host_data, rows * cols * sizeof(T),
                            cudaMemcpyHostToDevice));
    }
  } else {
    data = host_data;
  }
}

template <typename T>
void Matrix<T>::resize(int rows, int cols) {
  if (cols != this->cols) {
    throw std::logic_error(
        "changing number of columns in Matrix::resize is not implemented yet");
  }
  if (rows < this->rows) {
    throw std::logic_error(
        "reducing number of rows in Matrix::resize is not implemented yet");
  }
  auto new_storage =
      new rmm::device_uvector<T>(rows * cols, rmm::cuda_stream_view());
  CHECK_CUDA(cudaMemcpy(new_storage->data(), data,
                        this->rows * this->cols * sizeof(T),
                        cudaMemcpyDeviceToDevice));
  int extra_rows = rows - this->rows;
  CHECK_CUDA(cudaMemset(new_storage->data() + this->rows * this->cols, 0,
                        extra_rows * cols * sizeof(T)));
  storage.reset(new_storage);
  data = storage->data();
  this->rows = rows;
  this->cols = cols;
}

template <typename T>
void Matrix<T>::assign_rows(const Vector<int> &rowids, const Matrix<T> &other) {
  if (other.cols != cols) {
    throw std::invalid_argument(
        "column dimensionality mismatch in Matrix::assign_rows");
  }

  auto count = thrust::make_counting_iterator<int>(0);
  int other_cols = other.cols, other_rows = other.rows;

  int *rowids_data = rowids.data;
  T *other_data = other.data;
  T *self_data = data;

  thrust::for_each(count, count + (other_rows * other_cols),
                   [=] __device__(int i) {
                     int col = i % other_cols;
                     int row = rowids_data[i / other_cols];
                     int idx = col + row * other_cols;
                     self_data[idx] = other_data[i];
                   });
}

template <typename T>
__global__ void calculate_norms_kernel(const T *input, int rows, int cols,
                                       T *output) {
  static __shared__ float shared[32];
  for (int i = blockIdx.x; i < rows; i += gridDim.x) {
    float value = convert<T, float>(input[i * cols + threadIdx.x]);
    float squared_norm = dot(value, value, shared);
    if (threadIdx.x == 0) {
      float norm = sqrt(squared_norm);
      if (norm == 0) {
        norm = 1e-10;
      }
      output[i] = convert<float, T>(norm);
    }
  }
}

template <typename T>
Matrix<T> Matrix<T>::calculate_norms() const {
  int devId;
  CHECK_CUDA(cudaGetDevice(&devId));

  int multiprocessor_count;
  CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count,
                                    cudaDevAttrMultiProcessorCount, devId));

  int block_count = 256 * multiprocessor_count;
  int thread_count = cols;

  Matrix<T> output(1, rows, NULL);
  calculate_norms_kernel<<<block_count, thread_count>>>(
      data, rows, cols, output.data);

  CHECK_CUDA(cudaDeviceSynchronize());
  return output;
}

template <typename T>
void Matrix<T>::to_host(T *out) const {
  CHECK_CUDA(cudaMemcpy(out, data, rows * cols * sizeof(T),
                        cudaMemcpyDeviceToHost));
}

template struct Matrix<float>;
template struct Matrix<half>;

CSRMatrix::CSRMatrix(int rows, int cols, int nonzeros, const int *indptr_,
                     const int *indices_, const float *data_)
    : rows(rows), cols(cols), nonzeros(nonzeros) {

  CHECK_CUDA(cudaMalloc(&indptr, (rows + 1) * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(indptr, indptr_, (rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&indices, nonzeros * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(indices, indices_, nonzeros * sizeof(int),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&data, nonzeros * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(data, data_, nonzeros * sizeof(int), cudaMemcpyHostToDevice));
}

CSRMatrix::~CSRMatrix() {
  CHECK_CUDA(cudaFree(indices));
  CHECK_CUDA(cudaFree(indptr));
  CHECK_CUDA(cudaFree(data));
}

COOMatrix::COOMatrix(int rows, int cols, int nonzeros, const int *row_,
                     const int *col_, const float *data_)
    : rows(rows), cols(cols), nonzeros(nonzeros) {

  CHECK_CUDA(cudaMalloc(&row, nonzeros * sizeof(int)));
  CHECK_CUDA(
      cudaMemcpy(row, row_, nonzeros * sizeof(int), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&col, nonzeros * sizeof(int)));
  CHECK_CUDA(
      cudaMemcpy(col, col_, nonzeros * sizeof(int), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&data, nonzeros * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(data, data_, nonzeros * sizeof(int), cudaMemcpyHostToDevice));
}

COOMatrix::~COOMatrix() {
  CHECK_CUDA(cudaFree(row));
  CHECK_CUDA(cudaFree(col));
  CHECK_CUDA(cudaFree(data));
}
} // namespace gpu
} // namespace implicit
