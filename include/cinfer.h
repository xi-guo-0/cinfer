#ifndef CINFER_H_
#define CINFER_H_

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stddef.h>
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CINFER_FLOAT32,
} CINFER_NUMERIC_TYPE;

typedef struct {
  void *data;
  CINFER_NUMERIC_TYPE type;
  size_t *shape;
  size_t dim;
} Tensor;

Tensor *create_tensor(CINFER_NUMERIC_TYPE type, size_t *shape, size_t dim);

void free_tensor(Tensor *tensor);

void tensor_add(Tensor *a, Tensor *b, Tensor *result);
void tensor_matmul(Tensor *a, Tensor *b, Tensor *result);
void tensor_multiply(Tensor *a, Tensor *b, Tensor *result);
void tensor_relu(Tensor *input, Tensor *output);
void tensor_sigmoid(Tensor *input, Tensor *output);

#ifdef __cplusplus
}
#endif

#endif // CINFER_H_
