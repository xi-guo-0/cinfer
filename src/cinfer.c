#include "cinfer.h"
#include "cJSON.h"
#include "cblas.h"
#include <stdlib.h>
#include <string.h>

Tensor *create_tensor(CINFER_NUMERIC_TYPE type, size_t *shape, size_t dim) {
  Tensor *tensor = (Tensor *)calloc(1, sizeof(Tensor));
  if (tensor == NULL) {
    return NULL;
  }

  tensor->type = type;
  tensor->dim = dim;

  tensor->shape = (size_t *)calloc(dim, sizeof(size_t));
  if (tensor->shape == NULL) {
    free(tensor);
    return NULL;
  }

  for (int i = 0; i < dim; i++) {
    tensor->shape[i] = shape[i];
  }

  size_t total_size = 1;
  for (int i = 0; i < dim; i++) {
    total_size *= shape[i];
  }

  tensor->data = calloc(total_size, sizeof(float));

  if (tensor->data == NULL) {
    free(tensor->shape);
    free(tensor);
    return NULL;
  }

  return tensor;
}

void free_tensor(Tensor *tensor) {
  if (tensor != NULL) {
    if (tensor->data != NULL) {
      free(tensor->data);
    }
    if (tensor->shape != NULL) {
      free(tensor->shape);
    }
    free(tensor);
  }
}

void tensor_add(Tensor *a, Tensor *b, Tensor *result) {
  size_t size_a = 1;
  for (size_t i = 0; i < a->dim; i++) {
    size_a *= a->shape[i];
  }

  size_t size_b = 1;
  for (size_t i = 0; i < b->dim; i++) {
    size_b *= b->shape[i];
  }
  cblas_scopy(size_a, (float *)a->data, 1, (float *)result->data, 1);
  cblas_saxpy(size_b, 1.0f, (float *)b->data, 1, (float *)result->data, 1);
}

void tensor_matmul(Tensor *a, Tensor *b, Tensor *result) {
  int m = a->shape[0];
  int k = a->shape[1];
  int n = b->shape[1];

  float alpha = 1.0f;
  float beta = 0.0f;

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
              (float *)a->data, k, (float *)b->data, n, beta,
              (float *)result->data, n);
}

void tensor_multiply(Tensor *a, Tensor *b, Tensor *result) {
  size_t size = 1;
  for (size_t i = 0; i < a->dim; i++) {
    size *= a->shape[i];
  }

  float *a_data = (float *)a->data;
  float *b_data = (float *)b->data;
  float *result_data = (float *)result->data;

  for (size_t i = 0; i < size; i++) {
    result_data[i] = a_data[i] * b_data[i];
  }
}

void tensor_relu(Tensor *input, Tensor *output) {
  size_t size = 1;
  for (size_t i = 0; i < input->dim; i++) {
    size *= input->shape[i];
  }

  float *input_data = (float *)input->data;
  float *output_data = (float *)output->data;

  for (size_t i = 0; i < size; i++) {
    output_data[i] = input_data[i] > 0 ? input_data[i] : 0;
  }
}
