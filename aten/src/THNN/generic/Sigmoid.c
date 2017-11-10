#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Sigmoid.c"
#else


// THNN 是由宏实现的 一个 类 #define THNN_(NAME) TH_CONCAT_3(THNN_, Real, NAME) 在 aten/src/THNN/THNN.h 中
// Sigmoid_updateOutput 为啥用 updateOutput 这个 词 呢？
// 这个编译的时候会变成  
// void THNN_RealSigmoid_updateOutput(THNNState *state, THTensor *input, THTensor *output);
// 这个才是函数的真面目！！！！！！！！！！！！！！！！
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
