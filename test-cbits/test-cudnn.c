#include "test-cudnn.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <cudnn.h>

int test_cudnn_binding_version(void) {
    return 1;
}

int test_cudnn_library_version(void) {
    return CUDNN_VERSION;
}

// #define DEBUG 1

/**
 * Relevant references:
 *
 * - https://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
 * - https://docs.nvidia.com/deeplearning/cudnn/backend/latest/index.html
 *   (in particular the `ops` and `cnn` libraries)
 * - https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
 * - https://cs231n.github.io/convolutional-networks/
 *
 * Note on kernels: the kernel should have as many channels as the input
 * ("input features"), but we can have multiple kernels ("output features").
 * The result will be of size batch size * output features * height * width.
 *
 * There is a helpful diagram at
 * https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#convo-intro
 * that illustrates this.
 */

#define checkCUDNN(expression)                                                 \
  {                                                                            \
    cudnnStatus_t status = (expression);                                       \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      printf("Error on line %d: %s\n", __LINE__, cudnnGetErrorString(status)); \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

float* test_cudnn_convolve(
  cudnnConvolutionMode_t mode,
  int vertical_stride, int horizontal_stride,
  int num_kernels, int kernel_height, int kernel_width,
  float* kernel,
  int num_images, int input_channels, int input_height, int input_width,
  float* input,
  int* output_height, int* output_width
) {
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

#ifdef DEBUG
    printf("mode = %d, vertical_stride = %d, horizontal_stride = %d\n", mode, vertical_stride, horizontal_stride);
    printf("num_kernels = %d, kernel_height = %d, kernel_width = %d\n", num_kernels, kernel_height, kernel_width);
    printf("num_images = %d, input_channels = %d, input_height = %d, input_width = %d\n", num_images, input_channels, input_height, input_width);
#endif

    /**
     * Configure convolution
     */

    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
      /* pad_h             */ 0,
      /* pad_w             */ 0,
      /* vertical stride   */ vertical_stride,
      /* horizontal stride */ horizontal_stride,
      /* dilation_h        */ 1, // No dilation
      /* dilation_w        */ 1,
      /* mode              */ mode,
      /* computeType       */ CUDNN_DATA_FLOAT));

    /**
     * Setup input
     */

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
      /* format   */ CUDNN_TENSOR_NCHW,
      /* dataType */ CUDNN_DATA_FLOAT,
      /* n        */ num_images,
      /* c        */ input_channels,
      /* h        */ input_height,
      /* w        */ input_width));

    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
      /* dataType        */ CUDNN_DATA_FLOAT,
      /* format          */ CUDNN_TENSOR_NCHW,
      /* output channels */ num_kernels,
      /* input channels  */ input_channels,
      /* h               */ kernel_height,
      /* w               */ kernel_width));

    /**
     * Setup output
     */

    int num_output_images = -1;
    int output_channels   = -1;

    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
      /* convDesc        */ convolution_descriptor,
      /* inputTensorDesc */ input_descriptor,
      /* filterDesc      */ kernel_descriptor,
      /* n               */ &num_output_images,
      /* c               */ &output_channels,
      /* h               */ output_height,
      /* w               */ output_width));

#ifdef DEBUG
    printf("num_output_images = %d, output_channels = %d, output_height = %d, output_width = %d\n", num_output_images, output_channels, *output_height, *output_width);
#endif

    assert(num_output_images == num_images);
    assert(output_channels   == num_kernels);

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
      /* format   */ CUDNN_TENSOR_NCHW,
      /* dataType */ CUDNN_DATA_FLOAT,
      /* n        */ num_output_images,
      /* c        */ output_channels,
      /* h        */ *output_height,
      /* w        */ *output_width));

    /**
     * Prepare convolution
     */

    cudnnConvolutionFwdAlgoPerf_t convolution_algorithm_perf;
    int returned_algo_count;
    checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn,
      /* xDesc              */ input_descriptor,
      /* wDesc              */ kernel_descriptor,
      /* convDesc           */ convolution_descriptor,
      /* yDesc              */ output_descriptor,
      /* requestedAlgoCount */ 1,
      /* returnedAlgoCount  */ &returned_algo_count,
      /* perfResults        */ &convolution_algorithm_perf));
    cudnnConvolutionFwdAlgo_t convolution_algorithm = convolution_algorithm_perf.algo;

    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
      /* xDesc       */ input_descriptor,
      /* wDesc       */ kernel_descriptor,
      /* convDesc    */ convolution_descriptor,
      /* yDesc       */ output_descriptor,
      /* algo        */ convolution_algorithm,
      /* sizeInBytes */ &workspace_bytes));

    /**
     * Allocate device memory
     */

    int input_bytes  = num_images  * input_channels  * input_height  * input_width  * sizeof(float);
    int output_bytes = num_output_images * output_channels * (*output_height) * (*output_width) * sizeof(float);
    int kernel_bytes = num_kernels * input_channels * kernel_height * kernel_width * sizeof(float);

    void*  d_workspace = NULL;
    float* d_input     = NULL;
    float* d_output    = NULL;
    float* d_kernel    = NULL;

    cudaMalloc((void**) &d_workspace, workspace_bytes);
    cudaMalloc((void**) &d_input, input_bytes);
    cudaMalloc((void**) &d_output, output_bytes);
    cudaMalloc((void**) &d_kernel, kernel_bytes);

    /**
     * Initialize memory
     *
     * Everything up to this point has been completely independent from the
     * specific choice of input and kernel (apart from their size).
     */

    cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_bytes);

    /**
     * Execute the convolution
     */

    float alpha = 1, beta = 0; // no blending
    checkCUDNN(cudnnConvolutionForward(cudnn,
      /* alpha                */ &alpha,
      /* xDesc                */ input_descriptor,
      /* x                    */ d_input,
      /* wDesc                */ kernel_descriptor,
      /* w                    */ d_kernel,
      /* convDesc             */ convolution_descriptor,
      /* algo                 */ convolution_algorithm,
      /* workSpace            */ d_workspace,
      /* workSpaceSizeInBytes */ workspace_bytes,
      /* beta                 */ &beta,
      /* yDesc                */ output_descriptor,
      /* y                    */ d_output));

    /**
     * Copy results back to host and deallocate resources
     */

    float* output = (float*) malloc(output_bytes);
    cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_workspace);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);

    checkCUDNN(cudnnDestroy(cudnn));

    return output;
}
