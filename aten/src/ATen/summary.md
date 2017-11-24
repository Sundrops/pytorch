# 几个类型总结

## Tenosr TensorBase TensorImpl Type Storage

* Type: 表示数据类型
* Storage ： 存数据的地方
* TensorImpl : 有一个 引用计数
* TensorBase : 操作 TensorImpl 的引用计数
* Tensor : 上层建筑

Tensor 继承 TensorBase; TensorBase 包含 TensorImpl; TensorImpl 包含 （Type, THFloatTensor); THFLoatTensor 指向 Storage！！！！！
Type 中包含 Context* 全局唯一： Context 中注册这 所有的 Type。Type 中包含了这个 Type 的Tensor能用的 方法。


**Type可以**：

* 根据 *data 创建 storage 和 Tensor！！
* type 如何创建的 Tensor 嘞。


```c++
struct CPUFloatTensor final : public TensorImpl {
public:
  explicit CPUFloatTensor(Context* context);
  CPUFloatTensor(Context* context, THFloatTensor * tensor);
  virtual ~CPUFloatTensor();
  virtual const char * toString() const override;
  virtual IntList sizes() const override;
  virtual IntList strides() const override;
  virtual int64_t dim() const override;
  virtual Scalar localScalar() override;
  virtual void assign_(Scalar s) override;
  virtual void * unsafeGetTH(bool retain) override;
  virtual std::unique_ptr<Storage> storage() override;
  static const char * typeString();

//TODO(zach): sort of friend permissions later so this
// can be protected
public:
  THFloatTensor * tensor; 
  Context* context; // 系统环境
  friend struct CPUFloatType; // 类型创建 对应Tensor
};
```


## Context
包含了什么属性：

* type_registry : 当前系统，所有的类型都注册在这里。（（这里面的值是由 Type 给注册进去的）
* generator：当前系统，所有 generator 都注册在这里
* THCState : 用来记录当前的 状态

这个类看样子是 用来表示当前 系统的 的环境的。是单例 的。

## Backend

用来表示后端的
```c++
enum class Backend {
  CPU,
  CUDA,
  SparseCPU,
  SparseCUDA,
  Undefined,
  NumOptions
};
```
## Generator
这个应该是随机数生成器。