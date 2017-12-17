# ATEN 阅读笔记

**THNN/init.c** 里面有很多 `Check` 宏定义。


## TH

* `#define THTensor          TH_CONCAT_3(TH,Real,Tensor) ` 生成Token `THRealTensor`
* `#define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)` 生成 Token `THRealTensor_NAME`

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


