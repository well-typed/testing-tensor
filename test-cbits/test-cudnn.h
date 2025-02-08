#pragma once

#include "cudnn.h"

int test_cudnn_binding_version(void);
int test_cudnn_library_version(void);

/**
 * Convolution
 *
 * NOTE: This is just for testing purposes, and we take various shortcuts.
 * Not intended for production use.
 */
float* test_cudnn_convolve(
  cudnnConvolutionMode_t mode,
  int vertical_stride, int horizontal_stride,
  int num_kernels, int kernel_height, int kernel_width,
  float* kernel,
  int num_images, int input_channels, int input_height, int input_width,
  float* input,
  int* output_height, int* output_width
);
