# 理解 pytorch 的前向与反向过程

要想理解 pytorch 前向与反向过程，需要从以下几个方面考虑

## 前向：

1. Variable 在前向过程中充当什么角色
2. Function 在前向过程中充当什么角色

## 反向：

1. pytorch 什么时候创建的 反向传导图，如何创建的
2. Variable 的梯度怎么处理的， 因为一个 Variable 可能有多个方向 传回来的梯度


