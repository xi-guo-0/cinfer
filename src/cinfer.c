#include "cinfer.h"
#include "cJSON.h"
#include "cblas.h"
#include <math.h>
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

  for (size_t i = 0; i < dim; i++) {
    tensor->shape[i] = shape[i];
  }

  size_t total_size = 1;
  for (size_t i = 0; i < dim; i++) {
    total_size *= shape[i];
  }

  tensor->data = (float *)calloc(total_size, sizeof(float));

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

void tensor_sigmoid(Tensor *input, Tensor *output) {
  size_t size = 1;
  for (size_t i = 0; i < input->dim; i++) {
    size *= input->shape[i];
  }

  float *input_data = (float *)input->data;
  float *output_data = (float *)output->data;

  for (size_t i = 0; i < size; i++) {
    output_data[i] = 1.0f / (1.0f + exp(-input_data[i]));
  }
}

Dense_Layer *create_dense_layer(size_t input_size, size_t output_size,
                                void (*activation)(Tensor *, Tensor *)) {
  Dense_Layer *layer = (Dense_Layer *)malloc(sizeof(Dense_Layer));
  if (layer == NULL)
    return NULL;
  size_t weight_shape[] = {input_size, output_size};
  layer->weights = create_tensor(CINFER_FLOAT32, weight_shape, 2);
  size_t bias_shape[] = {output_size};
  layer->bias = create_tensor(CINFER_FLOAT32, bias_shape, 1);
  layer->activation = activation;
  return layer;
}

void free_dense_layer(Dense_Layer *layer) {
  if (layer != NULL) {
    free_tensor(layer->weights);
    free_tensor(layer->bias);
    free(layer);
  }
}

void dense_layer_forward(Dense_Layer *layer, Tensor *input, Tensor *output) {
  size_t temp_shape[] = {input->shape[0], layer->weights->shape[1]};
  Tensor *temp = create_tensor(CINFER_FLOAT32, temp_shape, 2);
  tensor_matmul(input, layer->weights, temp);
  tensor_add(temp, layer->bias, output);
  if (layer->activation != NULL) {
    layer->activation(output, output);
  }
  free_tensor(temp);
}

Conv_Layer *create_conv_layer(size_t in_channels, size_t out_channels,
                              size_t kernel_size, size_t stride,
                              size_t padding) {
  Conv_Layer *layer = (Conv_Layer *)malloc(sizeof(Conv_Layer));
  if (layer == NULL)
    return NULL;
  size_t kernel_shape[] = {out_channels, in_channels, kernel_size, kernel_size};
  layer->kernels = create_tensor(CINFER_FLOAT32, kernel_shape, 4);
  if (layer->kernels == NULL) {
    free(layer);
    return NULL;
  }
  size_t bias_shape[] = {out_channels};
  layer->bias = create_tensor(CINFER_FLOAT32, bias_shape, 1);
  if (layer->bias == NULL) {
    free_tensor(layer->kernels);
    free(layer);
    return NULL;
  }
  layer->stride = stride;
  layer->padding = padding;
  return layer;
}

void free_conv_layer(Conv_Layer *layer) {
  if (layer != NULL) {
    free_tensor(layer->kernels);
    free_tensor(layer->bias);
    free(layer);
  }
}

void conv_layer_forward(Conv_Layer *layer, Tensor *input, Tensor *output) {
  if (layer == NULL || input == NULL || output == NULL)
    return;

  float *input_data = (float *)input->data;
  float *output_data = (float *)output->data;
  float *kernels_data = (float *)layer->kernels->data;
  float *bias_data = (float *)layer->bias->data;

  size_t input_height = input->shape[1];
  size_t input_width = input->shape[2];
  size_t input_channels = input->shape[0];

  size_t kernel_count = layer->kernels->shape[0];
  size_t kernel_size = layer->kernels->shape[2];
  size_t stride = layer->stride;
  size_t padding = layer->padding;

  size_t output_height =
      (input_height + 2 * padding - kernel_size) / stride + 1;
  size_t output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

  if (output->shape != NULL) {
    free(output->shape);
  }
  output->shape = (size_t *)malloc(3 * sizeof(size_t));
  if (output->shape == NULL) {
    return;
  }

  output->shape[0] = kernel_count;
  output->shape[1] = output_height;
  output->shape[2] = output_width;
  output->dim = 3;

  if (output->data == NULL) {
    output->data = calloc(
        output->shape[0] * output->shape[1] * output->shape[2], sizeof(float));
  }

  for (size_t oc = 0; oc < kernel_count; oc++) {
    for (size_t oh = 0; oh < output_height; oh++) {
      for (size_t ow = 0; ow < output_width; ow++) {
        float sum = bias_data[oc];
        for (size_t kh = 0; kh < kernel_size; kh++) {
          for (size_t kw = 0; kw < kernel_size; kw++) {
            size_t ih = oh * stride - padding + kh;
            size_t iw = ow * stride - padding + kw;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
              for (size_t ic = 0; ic < input_channels; ic++) {
                sum += input_data[ic * input_height * input_width +
                                  ih * input_width + iw] *
                       kernels_data[oc * input_channels * kernel_size *
                                        kernel_size +
                                    ic * kernel_size * kernel_size +
                                    kh * kernel_size + kw];
              }
            }
          }
        }

        output_data[oc * output_height * output_width + oh * output_width +
                    ow] = sum;
      }
    }
  }
}

Model *create_model(Layer **layers, size_t num_layers) {
  Model *model = (Model *)malloc(sizeof(Model));
  model->layers = layers;
  model->num_layers = num_layers;
  return model;
}

void free_model(Model *model) {
  for (size_t i = 0; i < model->num_layers; i++) {
    free(model->layers[i]->layer);
    free(model->layers[i]);
  }
  free(model->layers);
  free(model);
}

void model_forward(Model *model, Tensor *input, Tensor *output) {
  Tensor *current_input = input;
  Tensor *current_output = output;
  for (size_t i = 0; i < model->num_layers; i++) {
    Layer *layer = model->layers[i];
    switch (layer->type) {
    case LAYER_DENSE:
      dense_layer_forward((Dense_Layer *)layer->layer, current_input,
                          current_output);
      break;
    case LAYER_CONV:
      conv_layer_forward((Conv_Layer *)layer->layer, current_input,
                         current_output);
      break;
    }
    current_input = current_output;
  }
}
