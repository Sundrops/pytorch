# ATEN 阅读笔记

**THNN/init.c** 里面有很多 `Check` 宏定义。



## TH

* `#define THTensor          TH_CONCAT_3(TH,Real,Tensor) ` 生成Token `THRealTensor`
* `#define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)` 生成 Token `THRealTensor_NAME`

* `generic/THTensor.h` : 一些 shape，resize，创建操作。
* `THGeneral.h` : 一些 `Check` 操作。



```c
typedef struct THTensor
{
    int64_t *size; // 表示 shape
    int64_t *stride; // stride， 假设 size 为 [3, 2, 1, 4], 那么 stride 为 [ 8, 4, 4, 1], outmost 到 innermost 的步长。
                     // 如果 数据不是连续的， stride 就是另一种情况了。感觉 stride 就是为了 判断是否连续而生的。
    int nDimension; // 表示有几维

    // Note: storage->size may be greater than the recorded size
    // of a tensor
    THStorage *storage;
    ptrdiff_t storageOffset; // Tensor_data的起始地址 TENSOR_data = TENSOR->storage->data+TENSOR->storageOffset
    int refcount;

    char flag;

} THTensor;
```

```c
// 指向分配的空间。
typedef struct THStorage
{
    real *data;
    ptrdiff_t size; // 空间大小。
    // 引用计数
    int refcount;
    char flag;  // TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
    // 用来 给 此 Storage 分配空间的类
    THAllocator *allocator;
    void *allocatorContext;
    struct THStorage *view;
} THStorage;
```

## THNN

* `#define THNN_(NAME) TH_CONCAT_3(THNN_, Real, NAME) ` 生成 Token `THNN_RealNAME`
* `THNN_CHECK_SHAPE(I1, I2)` 检查 `I1,I2` 是不是形状相同。不相同会 报错。 



## THC


## THCUNN




## 关键类型总结

* Tensor
* TensorBase Tensor的一个基类，主要用来 处理 reference counting

```c++
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

  friend struct Type;

  //TODO(zach): sort out friend structes
public:
  // 一个 TensorImpl* 属性！！！！！！！！！！！
  TensorImpl * pImpl;
};
```

* TensorImpl : 实现特定类型需要继承的类 `CPUFloatTensor` 等等都得继承这个类。

```c++
struct TensorImpl {
  explicit TensorImpl(Type * type)
  :  refcount(1), is_scalar(false), type_(type) {}

  Type & type() const {
    return *type_;
  }
private:
  // TensorImpl 一个 refcount， 一个 is_scalar, 一个 type_
  std::atomic<int> refcount;
  bool is_scalar;
  Type * type_;
};
```

* Type : 

```c++
struct AT_API Type {
  explicit Type(Context * context)
  : context(context) {}
protected:
  // 只有一个 Context 属性
  Context* context;
```

* Context ： 进程期间就只有一个 Context 对象（单例模式）

```c++
class AT_API Context {
public:
  Context();
  // 获取Type 的工具方法
  // 根据 Backend 和 标量类型 获得 Type， Type 不就是 标量类型 加 Backend 吗
  Type & getType(Backend p, ScalarType s) {
    initCUDAIfNeeded(p);
    auto & type = type_registry[static_cast<int>(p)][static_cast<int>(s)];

    if(!type) {
      // there is only a single Undefined Type.
      if (p == Backend::Undefined || s == ScalarType::Undefined) {
        auto & undef = type_registry[static_cast<int>(Backend::Undefined)][static_cast<int>(ScalarType::Undefined)];
        if (undef) return *undef;
      }
      runtime_error("%s%sType is not enabled.",toString(p),toString(s));
    }
    return *type;
  }
  Generator & defaultGenerator(Backend p) {
    initCUDAIfNeeded(p);
    auto & generator = generator_registry[static_cast<int>(p)];
    if(!generator)
      runtime_error("%s backend type not enabled.",toString(p));
    return *generator;
  }
  bool hasCUDA() const;
  // defined in header so that getType has ability to inline
  // call_once check. getType is called fairly frequently
  THCState* lazyInitCUDA() {
    std::call_once(thc_init,[&] {
      doInitCUDA();
    });
    return thc_state;
  }
  ~Context();

  // 注册的 generator ？？？？
  // 一维数组： 生成器的数量
  std::unique_ptr<Generator>
    generator_registry[static_cast<int>(Backend::NumOptions)];

  // 系统的所有 Type 保存在这里。
  // type_registry[numBackend][numOptions]; 是个二维数组
  std::unique_ptr<Type> type_registry
    [static_cast<int>(Backend::NumOptions)]
    [static_cast<int>(ScalarType::NumOptions)];
  
  THCState * thc_state;
private:
  void initCUDAIfNeeded(Backend p) {
    if(p == Backend::CUDA)
      lazyInitCUDA();
  }
  void doInitCUDA();
  std::once_flag thc_init;
};

// 里面有一个静态 对象 Context。
AT_API Context & globalContext();

static inline void init() {
  globalContext();
}

// 通过 Backend 和 ScalarType 获取 Type。。。
static inline Type& getType(Backend p, ScalarType s) {
  return globalContext().getType(p,s);
}

// 用来获取类型，类型用来创建 Tensor， 666
static inline Type& CPU(ScalarType s) {
  return getType(Backend::CPU, s);
}

static inline Type& CUDA(ScalarType s) {
  return getType(Backend::CUDA, s);
}

static inline bool hasCUDA() {
  return globalContext().hasCUDA();
}
```

一般的创建 Tensor 过程：

```c++
Tensor d = CPU(kFloat).ones({3, 4}); 
// CPU(kFloat) 获取 Type 对象，然后由 Type 对象创建
// kFloat 是常量表达式。是个 ScalarType 对象。 在 ScalarType.
```


## ATEN 编译后

按照官网所述编译过程执行一遍，在 `ATen/build/src/ATen/Aten` 中会生成以下文件:
```
CUDAFloatStorage.cpp 
CUDAFloatStorage.h
CUDAFloatTensor.cpp
CUDAFloatTensor.h
CUDAFloatType.cpp
CUDAFloatType.h
```

**Tensor**

```c++
struct CPUFloatTensor final : public TensorImpl {
public:
  explicit CPUFloatTensor(Context* context);
  CPUFloatTensor(Context* context, THFloatTensor * tensor);
  virtual ~CPUFloatTensor();

public:
  THFloatTensor * tensor;
  Context* context;
  friend struct CPUFloatType;
};
```

**Storage**

```c++
struct CPUFloatStorage : public Storage {
public:
  explicit CPUFloatStorage(Context* context);
  CPUFloatStorage(Context* context, THFloatStorage *wrapped);
  CPUFloatStorage(Context* context, std::size_t size);
  CPUFloatStorage(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~CPUFloatStorage();
protected:
  friend struct CPUFloatType;
  THFloatStorage *storage;
  Context* context;
};
```

**Type**

```c++
// 这里面有 CPUFloatType 支持所有运算。
struct CPUFloatType final : public Type {
  explicit CPUFloatType(Context* context); 
```

```c++
struct CUDAFloatStorage : public Storage {
public:
  explicit CUDAFloatStorage(Context* context);
  CUDAFloatStorage(Context* context, THCudaStorage *wrapped);
  CUDAFloatStorage(Context* context, std::size_t size);
  CUDAFloatStorage(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~CUDAFloatStorage();
protected:
  friend struct CUDAFloatType;
  THCudaStorage *storage;
  Context* context;
};


struct CUDAFloatTensor final : public TensorImpl {
public:
  explicit CUDAFloatTensor(Context* context);
  CUDAFloatTensor(Context* context, THCudaTensor * tensor);

//TODO(zach): sort of friend permissions later so this
// can be protected
public:
  THCudaTensor * tensor;
  Context* context;
  friend struct CUDAFloatType;
};

struct CUDAFloatType final : public Type {
  explicit CUDAFloatType(Context* context);
```
