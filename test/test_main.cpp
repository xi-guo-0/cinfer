#include "cinfer.h"
#include "gtest/gtest.h"

TEST(TensorTest, CreateAndFreeTensor) {
  size_t shape[] = {2, 3};
  Tensor *tensor = create_tensor(CINFER_FLOAT32, shape, 2);

  ASSERT_NE(tensor, nullptr);
  ASSERT_EQ(tensor->type, CINFER_FLOAT32);
  ASSERT_EQ(tensor->dim, 2);
  ASSERT_EQ(tensor->shape[0], 2);
  ASSERT_EQ(tensor->shape[1], 3);

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
  ASSERT_FLOAT_EQ(result_data[0], 5.0f);
  ASSERT_FLOAT_EQ(result_data[1], 7.0f);
  ASSERT_FLOAT_EQ(result_data[2], 9.0f);

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
  ASSERT_FLOAT_EQ(result_data[0], 6.0f);
  ASSERT_FLOAT_EQ(result_data[1], 8.0f);
  ASSERT_FLOAT_EQ(result_data[2], 10.0f);
  ASSERT_FLOAT_EQ(result_data[3], 12.0f);

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
  ASSERT_FLOAT_EQ(result_data[0], 10.0f);
  ASSERT_FLOAT_EQ(result_data[1], 12.0f);
  ASSERT_FLOAT_EQ(result_data[2], 14.0f);
  ASSERT_FLOAT_EQ(result_data[3], 16.0f);
  ASSERT_FLOAT_EQ(result_data[4], 18.0f);
  ASSERT_FLOAT_EQ(result_data[5], 20.0f);
  ASSERT_FLOAT_EQ(result_data[6], 22.0f);
  ASSERT_FLOAT_EQ(result_data[7], 24.0f);

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
    ASSERT_FLOAT_EQ(result_data[i], (float)(i + i + 18));
  }

  free_tensor(a);
  free_tensor(b);
  free_tensor(result);
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
