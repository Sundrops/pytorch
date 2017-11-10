# THNN 中的 macro 总结

* #define THNN_(NAME) TH_CONCAT_3(THNN_, Real, NAME)
* THTensor_
* TH_TENSOR_APPLY3
* THNN_CHECK_NELEMENT
* #define THTensor          TH_CONCAT_3(TH,Real,Tensor)
* #define THTensor_(NAME)   TH_CONCAT_4(TH, Real, Tensor_, NAME)