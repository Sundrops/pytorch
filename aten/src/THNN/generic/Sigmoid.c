#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Sigmoid.c"
#else


// THNN 是由宏实现的 一个 类 #define THNN_(NAME) TH_CONCAT_3(THNN_, Real, NAME) 在 aten/src/THNN/THNN.h 中
// Sigmoid_updateOutput 为啥用 updateOutput 这个 词 呢？
// 这个编译的时候会变成  
// void THNN_RealSigmoid_updateOutput(THNNState *state, THTensor *input, THTensor *output);
// 这个才是函数的真面目！！！！！！！！！！！！！！！！
// 作用，给定 input ，求 output
void THNN_(Sigmoid_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  // #define THTensor_(NAME) TH_CONCAT_4(TH, Real, Tensor_, NAME), 
  // 这个在编译的时候，会变成 THRealTensor_sigmoid(output, input);
  // THNN_ 比 THTensor 是要多一个 THNNState 
  THTensor_(sigmoid)(output, input);
}

// 作用， 给定 output, gradOutput, 求 gradInput
void THNN_(Sigmoid_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output)
{
  THNN_CHECK_NELEMENT(output, gradOutput);
  THTensor_(resizeAs)(gradInput, output);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,
    real z = *output_data;
    *gradInput_data = *gradOutput_data * (1. - z) * z;
  );
}

#endif

/*
void THNN_(Sigmoid_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  // #define THTensor_(NAME) TH_CONCAT_4(TH, Real, Tensor_, NAME), 
  // 这个在编译的时候，会变成 THRealTensor_sigmoid(output, input);
  // THNN_ 比 THTensor 是要多一个 THNNState 
  THTensor_(sigmoid)(output, input);
}

被 预处理器操作完后应该是这样的
void THNN_RealSigmoid_updateOutput(THNNState *state, THTensor *input, THTensor *output){
  THRealTensor_sigmoid(output, input);
}
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
void THNN_(Sigmoid_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output)
{
  THNN_CHECK_NELEMENT(output, gradOutput);
  THTensor_(resizeAs)(gradInput, output);
  
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,
    real z = *output_data;
    *gradInput_data = *gradOutput_data * (1. - z) * z;
  );
}
被预处理器操作完后，是这样的
void THNN_RealSigmoid_updateGradInput(THNNState *state, THTensor *gradOutput, THTensor *gradInput, THTensor *output){
  // 先检查一下 output 和 gradOutput 的 元素是否匹配
  THRealTensor_resizeAs(gradInput, output); // 为啥要这样操作, 因为是 sigmoid 嘛， input 是要和 output 一样的咯！！！！！
  

}

#define TH_TENSOR_APPLY3_D(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, DIM, CODE) \
{ \
  int TH_TENSOR_APPLY_hasFinished = 0; \
  int64_t TH_TENSOR_dim_index = 0; \
  __TH_TENSOR_APPLYX_PREAMBLE(real, gradInput, -1, 1) \
  __TH_TENSOR_APPLYX_PREAMBLE(real, gradOutput, -1, 1) \
  __TH_TENSOR_APPLYX_PREAMBLE(real, output, -1, 1) \
  // 经过这里之后 ，gradInput 变成 gradInput_data ........ 
                                                                        \
  int elements_equal = 1;                                               \
  if(gradInput_n != gradOutput_n) {                                      \
    elements_equal = 0;                                                 \
  }                                                                     \
  else if(gradInput_n != output_n) {                                 \
    elements_equal = 0;                                                 \
  }                                                                     \
  if (elements_equal == 0) {                                            \
    THDescBuff T1buff = _THSizeDesc(gradInput->size, gradInput->nDimension); \
    THDescBuff T2buff = _THSizeDesc(TENSOR2->size, TENSOR2->nDimension); \
    THDescBuff T3buff = _THSizeDesc(TENSOR3->size, TENSOR3->nDimension); \
    THError("inconsistent tensor size, expected %s %s, %s %s and %s %s to have the same " \
            "number of elements, but got %d, %d and %d elements respectively", \
            #TENSOR1, T1buff.str, #TENSOR2, T2buff.str, #TENSOR3, T3buff.str, \
            TENSOR1##_n, TENSOR2##_n, TENSOR3##_n);                     \
  }                                                                     \
                                                                        \
  while(!TH_TENSOR_APPLY_hasFinished) \
  { \
    // Loop through the inner most region of the Tensor  \
    for(; gradInput_i < gradInput_size && TENSOR2##_i < TENSOR2##_size && TENSOR3##_i < TENSOR3##_size;\
     gradInput_i++, TENSOR2##_i++, TENSOR3##_i++, gradInput_data += gradInput_stride, TENSOR2##_data += TENSOR2##_stride, TENSOR3##_data += TENSOR3##_stride) \
    { \
      CODE \
    } \
    __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR1, 0) \
    __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR2, 0) \
    __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR3, 0) \
  } \
  if(TENSOR1##_counter != NULL) \
    THFree(TENSOR1##_counter); \
  if(TENSOR2##_counter != NULL) \
    THFree(TENSOR2##_counter); \
  if(TENSOR3##_counter != NULL) \
    THFree(TENSOR3##_counter); \
}
*/