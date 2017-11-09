# A TENsor library 总结

## ATen
* Context : 
* Type ： 只有一个 Context* 属性
* TensorImpl : `std::atomic<int> refcount; bool is_scalar; Type * type_;` 
* TensorBase : `TensorImpl * pImpl;` 用来管理 TensorImpl 的引用计数的
* Tensor : TensorBase 一个 子类， 仅仅是包装了一层 函数而已


## TH
