
#include <cstring>
#include <iostream>

#include "optimized/ref/depthwise_conv.hpp"
#include "BufferTensor.hpp"
#include "RamTensor.hpp"
#include "RomTensor.hpp"
#include "arenaAllocator.hpp"
#include "context.hpp"
#include "gtest/gtest.h"

#include "constants_convolution.hpp"
using std::cout;
using std::endl;

using namespace uTensor;

//#define DO_STRIDE_TESTS 1
/*********************************************
 * Generated Test number 
 *********************************************/
template <typename T>
void CalculateActivationRange(TfLiteFusedActivation activation,
                              T* activation_min, T* activation_max) {
  if (activation == kTfLiteActRelu) {
    *activation_min = 0;
    *activation_max = std::numeric_limits<T>::max();
  } else if (activation == kTfLiteActRelu6) {
    *activation_min = 0;
    *activation_max = 6;
  } else if (activation == kTfLiteActRelu1) {
    *activation_min = -1;
    *activation_max = 1;
  } else {
    *activation_min = std::numeric_limits<T>::lowest();
    *activation_max = std::numeric_limits<T>::max();
  }
}


TEST(Depthwise_Convolution, SimpleTest) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<25088*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  //const int input_elements = 12;
  //const int input_shape[] = {4, 1, 3, 2, 2};
  static const float input_values[12] = {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
  //NT: targeting 1 batch instead of 4; modifying dim(0)
  Tensor in = new RomTensor({ 1, 1, 3, 2, 2 } }, flt, input_values);

  //const int filter_elements = 16;
  //const int filter_shape[] = {4, 1, 2, 2, 4};
  static const float filter_values[] = {1, 2, 3, 4, -9, 10,  -11, 12,
                                 5, 6, 7, 8, 13, -14, 15,  -16};
  //NT: targeting 1 batch instead of 4; modifying dim(0)
  Tensor w = new RomTensor({1, 1, 2, 2, 4}, flt, filter_values);

  //const int bias_elements = 4;
  //const int bias_shape[] = {4, 1, 1, 1, 4};
  static const float bias_values[] = {1, 2, 3, 4};
  //NT: targeting 1 batch instead of 4; modifying dim(0)
  Tensor b = new RomTensor({1 1, 1, 1, 4}, flt, bias_values);

  //const int output_elements = 8;
  //const int output_shape[] = {4, 1, 2, 1, 4};
  //const int output_dims_count = 8;
  //float output_data[output_dims_count];
  //NT: targeting 1 batch instead of 4; modifying dim(0)
  Tensor out = new RamTensor({ 1, 1, 2, 1, 4 }, flt);

  DepthwiseConv::DepthwiseParams op_params;
  op_params.padding_type = 1;
  op_params.padding_values.width = 0;
  op_params.padding_values.height = 0;
  op_params.stride_width = 1;
  op_params.stride_height = 1;
  op_params.dilation_width_factor = 1;
  op_params.dilation_height_factor = 1;
  op_params.depth_multiplier = 2;
  op_params.float_activation_min: std::numeric_limits<T>::lowest();
  op_params.float_activation_max: std::numeric_limits<T>::max();

  const float golden[] = {
      71, -34, 99, -20, 91, -26, 127, -4,
  };

  DepthwiseConv<float> depth_conv;
  depth_conv.set_params(op_params);
  depth_conv.set_inputs({{ConvOperator<float>::in, in}, {ConvOperator<float>::filter, filter}, {ConvOperator<float>::bias, bias}})
            .set_outputs({ {ConvOperator<float>::out, out} })
            .eval();

  // Compare results
  //NT:  out for the night, TODO: the shape is output_shape[] = {4, 1, 2, 1, 4};
  //Ignore the batch
  //compare the values below
  TensorShape& out_shape = out->get_shape();
  for (int i = 0; i < out_shape[0]; i++) {
    for (int j = 0; j < out_shape[1]; j++) {
      size_t lin_index = j + i * c_shape[0];
      // Just need to cast the output
      EXPECT_EQ((uint8_t)c(i, j), s_c_ref[lin_index]);
    }
  }


  for(int i = 0; i < 25088; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_0_stride_1[i], 0.0001);
  }
}
