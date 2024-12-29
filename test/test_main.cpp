#include "cinfer.h"
#include "gtest/gtest.h"

TEST(TensorTest, CreateAndFreeTensor) {
  size_t shape[] = {2, 3};
  Tensor *tensor = create_tensor(CINFER_FLOAT32, shape, 2);

  ASSERT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->type, CINFER_FLOAT32);
  EXPECT_EQ(tensor->dim, 2);
  EXPECT_EQ(tensor->shape[0], 2);
  EXPECT_EQ(tensor->shape[1], 3);

  free_tensor(tensor);
}

TEST(TensorTest, TensorAdd1D) {
  size_t shape[] = {3};
  Tensor *a = create_tensor(CINFER_FLOAT32, shape, 1);
  Tensor *b = create_tensor(CINFER_FLOAT32, shape, 1);
  Tensor *result = create_tensor(CINFER_FLOAT32, shape, 1);

  float *a_data = (float *)a->data;
  float *b_data = (float *)b->data;
  a_data[0] = 1.0f;
  a_data[1] = 2.0f;
  a_data[2] = 3.0f;
  b_data[0] = 4.0f;
  b_data[1] = 5.0f;
  b_data[2] = 6.0f;

  tensor_add(a, b, result);

  float *result_data = (float *)result->data;
  EXPECT_FLOAT_EQ(result_data[0], 5.0f);
  EXPECT_FLOAT_EQ(result_data[1], 7.0f);
  EXPECT_FLOAT_EQ(result_data[2], 9.0f);

  free_tensor(a);
  free_tensor(b);
  free_tensor(result);
}

TEST(TensorTest, TensorAdd2D) {
  size_t shape[] = {2, 2};
  Tensor *a = create_tensor(CINFER_FLOAT32, shape, 2);
  Tensor *b = create_tensor(CINFER_FLOAT32, shape, 2);
  Tensor *result = create_tensor(CINFER_FLOAT32, shape, 2);

  float *a_data = (float *)a->data;
  float *b_data = (float *)b->data;
  a_data[0] = 1.0f;
  a_data[1] = 2.0f;
  a_data[2] = 3.0f;
  a_data[3] = 4.0f;
  b_data[0] = 5.0f;
  b_data[1] = 6.0f;
  b_data[2] = 7.0f;
  b_data[3] = 8.0f;

  tensor_add(a, b, result);

  float *result_data = (float *)result->data;
  EXPECT_FLOAT_EQ(result_data[0], 6.0f);
  EXPECT_FLOAT_EQ(result_data[1], 8.0f);
  EXPECT_FLOAT_EQ(result_data[2], 10.0f);
  EXPECT_FLOAT_EQ(result_data[3], 12.0f);

  free_tensor(a);
  free_tensor(b);
  free_tensor(result);
}

TEST(TensorTest, TensorAdd3D) {
  size_t shape[] = {2, 2, 2};
  Tensor *a = create_tensor(CINFER_FLOAT32, shape, 3);
  Tensor *b = create_tensor(CINFER_FLOAT32, shape, 3);
  Tensor *result = create_tensor(CINFER_FLOAT32, shape, 3);

  float *a_data = (float *)a->data;
  float *b_data = (float *)b->data;
  a_data[0] = 1.0f;
  a_data[1] = 2.0f;
  a_data[2] = 3.0f;
  a_data[3] = 4.0f;
  a_data[4] = 5.0f;
  a_data[5] = 6.0f;
  a_data[6] = 7.0f;
  a_data[7] = 8.0f;
  b_data[0] = 9.0f;
  b_data[1] = 10.0f;
  b_data[2] = 11.0f;
  b_data[3] = 12.0f;
  b_data[4] = 13.0f;
  b_data[5] = 14.0f;
  b_data[6] = 15.0f;
  b_data[7] = 16.0f;

  tensor_add(a, b, result);

  float *result_data = (float *)result->data;
  EXPECT_FLOAT_EQ(result_data[0], 10.0f);
  EXPECT_FLOAT_EQ(result_data[1], 12.0f);
  EXPECT_FLOAT_EQ(result_data[2], 14.0f);
  EXPECT_FLOAT_EQ(result_data[3], 16.0f);
  EXPECT_FLOAT_EQ(result_data[4], 18.0f);
  EXPECT_FLOAT_EQ(result_data[5], 20.0f);
  EXPECT_FLOAT_EQ(result_data[6], 22.0f);
  EXPECT_FLOAT_EQ(result_data[7], 24.0f);

  free_tensor(a);
  free_tensor(b);
  free_tensor(result);
}

TEST(TensorTest, TensorAdd4D) {
  size_t shape[] = {2, 2, 2, 2};
  Tensor *a = create_tensor(CINFER_FLOAT32, shape, 4);
  Tensor *b = create_tensor(CINFER_FLOAT32, shape, 4);
  Tensor *result = create_tensor(CINFER_FLOAT32, shape, 4);

  float *a_data = (float *)a->data;
  float *b_data = (float *)b->data;
  for (int i = 0; i < 16; i++) {
    a_data[i] = (float)(i + 1);
    b_data[i] = (float)(i + 17);
  }

  tensor_add(a, b, result);

  float *result_data = (float *)result->data;
  for (int i = 0; i < 16; i++) {
    EXPECT_FLOAT_EQ(result_data[i], (float)(i + i + 18));
  }

  free_tensor(a);
  free_tensor(b);
  free_tensor(result);
}

TEST(TensorTest, MatrixMultiplication) {
  size_t shape_a[] = {2, 3};
  size_t shape_b[] = {3, 2};
  size_t shape_result[] = {2, 2};

  Tensor *a = create_tensor(CINFER_FLOAT32, shape_a, 2);
  Tensor *b = create_tensor(CINFER_FLOAT32, shape_b, 2);
  Tensor *result = create_tensor(CINFER_FLOAT32, shape_result, 2);

  float *a_data = (float *)a->data;
  float *b_data = (float *)b->data;

  a_data[0] = 1.0f;
  a_data[1] = 2.0f;
  a_data[2] = 3.0f;
  a_data[3] = 4.0f;
  a_data[4] = 5.0f;
  a_data[5] = 6.0f;

  b_data[0] = 1.0f;
  b_data[1] = 2.0f;
  b_data[2] = 3.0f;
  b_data[3] = 4.0f;
  b_data[4] = 5.0f;
  b_data[5] = 6.0f;

  tensor_matmul(a, b, result);

  float *result_data = (float *)result->data;

  EXPECT_FLOAT_EQ(result_data[0], 22.0f);
  EXPECT_FLOAT_EQ(result_data[1], 28.0f);
  EXPECT_FLOAT_EQ(result_data[2], 49.0f);
  EXPECT_FLOAT_EQ(result_data[3], 64.0f);

  free_tensor(a);
  free_tensor(b);
  free_tensor(result);
}

TEST(TensorTest, TensorMultiply) {
  size_t shape[] = {2, 2};
  Tensor *a = create_tensor(CINFER_FLOAT32, shape, 2);
  Tensor *b = create_tensor(CINFER_FLOAT32, shape, 2);
  Tensor *result = create_tensor(CINFER_FLOAT32, shape, 2);

  float *a_data = (float *)a->data;
  float *b_data = (float *)b->data;
  a_data[0] = 1.0f;
  a_data[1] = 2.0f;
  a_data[2] = 3.0f;
  a_data[3] = 4.0f;
  b_data[0] = 5.0f;
  b_data[1] = 6.0f;
  b_data[2] = 7.0f;
  b_data[3] = 8.0f;

  tensor_multiply(a, b, result);

  float *result_data = (float *)result->data;
  EXPECT_FLOAT_EQ(result_data[0], 5.0f);
  EXPECT_FLOAT_EQ(result_data[1], 12.0f);
  EXPECT_FLOAT_EQ(result_data[2], 21.0f);
  EXPECT_FLOAT_EQ(result_data[3], 32.0f);

  free_tensor(a);
  free_tensor(b);
  free_tensor(result);
}

TEST(TensorTest, ReLUActivation) {
  size_t shape[] = {2, 2};
  Tensor *input = create_tensor(CINFER_FLOAT32, shape, 2);
  Tensor *output = create_tensor(CINFER_FLOAT32, shape, 2);

  float *input_data = (float *)input->data;
  input_data[0] = -1.0f;
  input_data[1] = 2.0f;
  input_data[2] = 0.0f;
  input_data[3] = 3.0f;

  tensor_relu(input, output);

  float *output_data = (float *)output->data;
  EXPECT_FLOAT_EQ(output_data[0], 0.0f);
  EXPECT_FLOAT_EQ(output_data[1], 2.0f);
  EXPECT_FLOAT_EQ(output_data[2], 0.0f);
  EXPECT_FLOAT_EQ(output_data[3], 3.0f);

  free_tensor(input);
  free_tensor(output);
}

TEST(TensorTest, SigmoidActivation) {
  size_t shape[] = {2, 2};
  Tensor *input = create_tensor(CINFER_FLOAT32, shape, 2);
  Tensor *output = create_tensor(CINFER_FLOAT32, shape, 2);

  float *input_data = (float *)input->data;
  input_data[0] = 0.0f;
  input_data[1] = 1.0f;
  input_data[2] = -1.0f;
  input_data[3] = 2.0f;

  tensor_sigmoid(input, output);

  float *output_data = (float *)output->data;
  ASSERT_NEAR(output_data[0], 0.5f, 1e-6);
  ASSERT_NEAR(output_data[1], 0.731059f, 1e-6);
  ASSERT_NEAR(output_data[2], 0.268941f, 1e-6);
  ASSERT_NEAR(output_data[3], 0.880797f, 1e-6);

  free_tensor(input);
  free_tensor(output);
}

TEST(LayerTest, DenseLayerForward) {
  Dense_Layer *layer = create_dense_layer(2, 2, tensor_relu);
  ASSERT_NE(layer, nullptr);

  float *weights_data = (float *)layer->weights->data;
  float *bias_data = (float *)layer->bias->data;

  weights_data[0] = 1.0f;
  weights_data[1] = 2.0f;
  weights_data[2] = 3.0f;
  weights_data[3] = 4.0f;

  bias_data[0] = 0.1f;
  bias_data[1] = 0.2f;

  size_t input_shape[] = {1, 2};
  Tensor *input = create_tensor(CINFER_FLOAT32, input_shape, 2);
  float *input_data = (float *)input->data;
  input_data[0] = 0.5f;
  input_data[1] = 1.5f;

  size_t output_shape[] = {1, 2};
  Tensor *output = create_tensor(CINFER_FLOAT32, output_shape, 2);

  dense_layer_forward(layer, input, output);

  float *output_data = (float *)output->data;
  ASSERT_NEAR(output_data[0], 5.1f, 1e-6);
  ASSERT_NEAR(output_data[1], 7.2f, 1e-6);

  free_tensor(input);
  free_tensor(output);
  free_dense_layer(layer);
}

TEST(LayerTest, ConvLayerForward) {
  size_t in_channels = 1;
  size_t out_channels = 1;
  size_t kernel_size = 3;
  size_t stride = 1;
  size_t padding = 0;
  Conv_Layer *layer = create_conv_layer(in_channels, out_channels, kernel_size,
                                        stride, padding);
  ASSERT_NE(layer, nullptr);

  float *kernels_data = (float *)layer->kernels->data;
  kernels_data[0] = 1.0f;
  kernels_data[1] = 0.0f;
  kernels_data[2] = -1.0f;
  kernels_data[3] = 1.0f;
  kernels_data[4] = 0.0f;
  kernels_data[5] = -1.0f;
  kernels_data[6] = 1.0f;
  kernels_data[7] = 0.0f;
  kernels_data[8] = -1.0f;

  float *bias_data = (float *)layer->bias->data;
  bias_data[0] = 0.1f;

  size_t input_shape[] = {1, 4, 4};
  Tensor *input = create_tensor(CINFER_FLOAT32, input_shape, 3);
  float *input_data = (float *)input->data;
  for (size_t i = 0; i < 16; i++) {
    input_data[i] = (float)i;
  }

  size_t output_shape[] = {1, 2, 2};
  Tensor *output = create_tensor(CINFER_FLOAT32, output_shape, 3);

  conv_layer_forward(layer, input, output);

  float *output_data = (float *)output->data;
  ASSERT_NEAR(output_data[0], -5.9f, 1e-6);
  ASSERT_NEAR(output_data[1], -5.9f, 1e-6);
  ASSERT_NEAR(output_data[2], -5.9f, 1e-6);
  ASSERT_NEAR(output_data[3], -5.9f, 1e-6);

  free_tensor(input);
  free_tensor(output);
  free_conv_layer(layer);
}

TEST(LayerTest, ConvLayerForwardBasic) {
  Conv_Layer *layer = create_conv_layer(1, 1, 3, 1, 0);
  size_t input_shape[] = {1, 5, 5};
  Tensor *input = create_tensor(CINFER_FLOAT32, input_shape, 3);
  Tensor *output = create_tensor(CINFER_FLOAT32, nullptr, 0);

  float *input_data = (float *)input->data;
  for (size_t i = 0; i < 25; ++i) {
    input_data[i] = 1.0f;
  }

  conv_layer_forward(layer, input, output);

  ASSERT_NE(output->data, nullptr);
  EXPECT_EQ(output->shape[0], 1);
  EXPECT_EQ(output->shape[1], 3);
  EXPECT_EQ(output->shape[2], 3);

  free_tensor(input);
  free_tensor(output);
  free_conv_layer(layer);
}

TEST(LayerTest, ConvLayerForwardNullInput) {
  Conv_Layer *layer = nullptr;
  Tensor *input = nullptr;
  Tensor *output = nullptr;

  conv_layer_forward(layer, input, output);
}

TEST(LayerTest, ConvLayerForwardStrideAndPadding) {
  Conv_Layer *layer = create_conv_layer(1, 1, 3, 2, 1);
  size_t input_shape[] = {1, 5, 5};
  Tensor *input = create_tensor(CINFER_FLOAT32, input_shape, 3);
  Tensor *output = create_tensor(CINFER_FLOAT32, nullptr, 0);

  float *input_data = (float *)input->data;
  for (size_t i = 0; i < 25; ++i) {
    input_data[i] = 1.0f;
  }

  conv_layer_forward(layer, input, output);

  ASSERT_NE(output->data, nullptr);
  EXPECT_EQ(output->shape[0], 1);
  EXPECT_EQ(output->shape[1], 3);
  EXPECT_EQ(output->shape[2], 3);

  free_tensor(input);
  free_tensor(output);
  free_conv_layer(layer);
}

TEST(LayerTest, ConvLayerForwardVariousKernelSizes) {
  Conv_Layer *layer = create_conv_layer(1, 1, 3, 1, 1);
  Tensor *input = create_tensor(CINFER_FLOAT32, (size_t[]){1, 5, 5}, 3);
  Tensor *output = create_tensor(CINFER_FLOAT32, (size_t[]){1, 5, 5}, 3);

  float *input_data = (float *)input->data;
  for (int i = 0; i < 25; i++) {
    input_data[i] = (float)i;
  }

  conv_layer_forward(layer, input, output);

  EXPECT_EQ(output->shape[0], 1);
  EXPECT_EQ(output->shape[1], 5);
  EXPECT_EQ(output->shape[2], 5);

  free_tensor(input);
  free_tensor(output);
  free_conv_layer(layer);
}

TEST(LayerTest, ConvLayerForwardDifferentInputShapes) {
  Conv_Layer *layer = create_conv_layer(1, 1, 3, 1, 1);
  Tensor *input = create_tensor(CINFER_FLOAT32, (size_t[]){1, 6, 6}, 3);
  Tensor *output = create_tensor(CINFER_FLOAT32, (size_t[]){1, 4, 4}, 3);

  float *input_data = (float *)input->data;
  for (int i = 0; i < 36; i++) {
    input_data[i] = (float)i;
  }

  conv_layer_forward(layer, input, output);

  EXPECT_EQ(output->shape[0], 1);
  EXPECT_EQ(output->shape[1], 6);
  EXPECT_EQ(output->shape[2], 6);

  free_tensor(input);
  free_tensor(output);
  free_conv_layer(layer);
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
