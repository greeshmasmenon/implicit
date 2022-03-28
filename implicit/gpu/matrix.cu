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

Matrix::Matrix(const Matrix &other, int rowid)
    : rows(1), cols(other.cols), data(other.at(rowid * other.cols)),
      storage(other.storage), itemsize(other.itemsize) {
  if (rowid >= other.rows) {
    throw std::invalid_argument("row index out of bounds for matrix");
  }
}

Matrix::Matrix(const Matrix &other, int start_rowid, int end_rowid)
    : rows(end_rowid - start_rowid), cols(other.cols),
      data(other.at(start_rowid * other.cols)), storage(other.storage),
      itemsize(other.itemsize) {
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

Matrix::Matrix(const Matrix &other, const Vector<int> &rowids)
    : rows(rowids.size), cols(other.cols), itemsize(other.itemsize) {
  storage.reset(
      new rmm::device_buffer(itemsize * rows * cols, rmm::cuda_stream_view()));
  data = storage->data();
  // TODO:
  if (itemsize == 4) {
    copy_rowids<float>(other, rowids.data, rows, cols, *this);
  } else {
    throw std::runtime_error("unknown itemsize initializing Matrix");
  }
}

Matrix::Matrix(int rows, int cols, void *host_data, bool allocate, int itemsize)
    : rows(rows), cols(cols), itemsize(itemsize) {
  if (allocate) {
    storage.reset(new rmm::device_buffer(itemsize * rows * cols,
                                         rmm::cuda_stream_view()));
    data = storage->data();
    if (host_data) {
      CHECK_CUDA(cudaMemcpy(data, host_data, rows * cols * itemsize,
                            cudaMemcpyHostToDevice));
    }
  } else {
    data = host_data;
  }
}

void Matrix::resize(int rows, int cols) {
  if (cols != this->cols) {
    throw std::logic_error(
        "changing number of columns in Matrix::resize is not implemented yet");
  }
  if (rows < this->rows) {
    throw std::logic_error(
        "reducing number of rows in Matrix::resize is not implemented yet");
  }
  auto new_storage =
      new rmm::device_buffer(itemsize * rows * cols, rmm::cuda_stream_view());
  CHECK_CUDA(cudaMemcpy(new_storage->data(), data,
                        this->rows * this->cols * itemsize,
                        cudaMemcpyDeviceToDevice));
  int extra_rows = rows - this->rows;
  storage.reset(new_storage);
  data = storage->data();
  CHECK_CUDA(
      cudaMemset(at(this->rows * this->cols), 0, extra_rows * cols * itemsize));

  this->rows = rows;
  this->cols = cols;
}

void Matrix::assign_rows(const Vector<int> &rowids, const Matrix &other) {
  if (other.cols != cols) {
    throw std::invalid_argument(
        "column dimensionality mismatch in Matrix::assign_rows");
  }

  auto count = thrust::make_counting_iterator<int>(0);
  int other_cols = other.cols, other_rows = other.rows;

  int *rowids_data = rowids.data;

  const float *other_data = other;
  float *self_data = *this;

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

Matrix Matrix::calculate_norms() const {
  int devId;
  CHECK_CUDA(cudaGetDevice(&devId));

  int multiprocessor_count;
  CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count,
                                    cudaDevAttrMultiProcessorCount, devId));

  int block_count = 256 * multiprocessor_count;
  int thread_count = cols;

  Matrix output(1, rows, NULL);

  if (itemsize == 4) {
    calculate_norms_kernel<float>
        <<<block_count, thread_count>>>(*this, rows, cols, output);
    // TODO  } else if (itemsize == 2) {
    //    calculate_norms_kernel<half><<<block_count, thread_count>>>(
    //        data, rows, cols, output.data);
  } else {
    throw std::runtime_error("unknown itemsize in calculate_norms");
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  return output;
}

void Matrix::to_host(void *out) const {
  CHECK_CUDA(
      cudaMemcpy(out, data, rows * cols * itemsize, cudaMemcpyDeviceToHost));
}

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
