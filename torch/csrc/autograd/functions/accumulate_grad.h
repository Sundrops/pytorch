#pragma once

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {


// 梯度的累加操作

struct AccumulateGrad : public Function {
  AccumulateGrad(Variable variable);

  virtual variable_list apply(const variable_list& inputs) override;

  Variable variable;
};

}}
