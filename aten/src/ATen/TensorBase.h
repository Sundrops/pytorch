#pragma once

#include "ATen/TensorImpl.h"
#include "ATen/UndefinedTensor.h"

namespace at { namespace detail {

// TensorBase is the base class for Tensor which handles the reference counting
// Tensor的一个基类，主要用来 处理 reference counting

struct TensorBase {
  TensorBase(): TensorBase(UndefinedTensor::singleton(), false) {}
  TensorBase(TensorImpl * self, bool retain)
  : pImpl(self) {
    if (pImpl == nullptr) {
      throw std::runtime_error("TensorBase with nullptr not supported");
    }
    if(retain && pImpl != UndefinedTensor::singleton())
      pImpl->retain();
  }
  TensorBase(const TensorBase & rhs)
  : pImpl(rhs.pImpl) {
    if (pImpl != UndefinedTensor::singleton())
      pImpl->retain();
  }
  TensorBase(TensorBase && rhs) noexcept
  : pImpl(rhs.pImpl) {
    rhs.pImpl = UndefinedTensor::singleton();
  }
  ~TensorBase() {
    if (pImpl != UndefinedTensor::singleton())
      pImpl->release();
  }

  // Operator= 都有这么多的 骚操作
  TensorBase & operator=(TensorBase && rhs) & {
    rhs.swap(*this);
    return *this;
  }
  TensorBase & operator=(TensorBase const & rhs) & {
      //TensorBase ctor retains original rhs.pImpl
      //then rhs.pImpl is swapped with this->pImpl
      //finally TensorBase dtor releases rhs.pImpl, which was originally this->pImpl
      TensorBase(rhs).swap(*this);
      return *this;
  }
  int64_t dim() const {
    return pImpl->dim();
  }

  // 两个 TensorBase 交换一下 指向的 TensorImpl
  void swap(TensorBase & rhs) {
    TensorImpl * tmp = pImpl;
    pImpl = rhs.pImpl;
    rhs.pImpl = tmp;
  }
  TensorImpl * get() const {
    return pImpl;
  }
  bool defined() const {
    return pImpl != UndefinedTensor::singleton();
  }

  friend struct Type;

  //TODO(zach): sort out friend structes
public:
  // 一个 TensorImpl* 属性！！！！！！！！！！！
  TensorImpl * pImpl;
};

}} // namespace at::detail
