#include <torch/extension.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;
typedef at::Half fp16;
typedef float fp32;

void cuda_forward_bf16_runner(int B, int T, int C, int H,  bf16 *r, bf16 *k, bf16 *v, float *w, bf16 *u, bf16 *y);
void cuda_forward_fp16_runner(int B, int T, int C, int H,  fp16 *r, fp16 *k, fp16 *v, float *w, fp16 *u, fp16 *y);
void cuda_forward_fp32_runner(int B, int T, int C, int H,  fp32 *r, fp32 *k, fp32 *v, float *w, fp32 *u, fp32 *y);

void forward_bf16_runner(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    cuda_forward_bf16_runner(B, T, C, H, r.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), w.data_ptr<float>(), u.data_ptr<bf16>(), y.data_ptr<bf16>());
}
void forward_fp16_runner(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    cuda_forward_fp16_runner(B, T, C, H, r.data_ptr<fp16>(), k.data_ptr<fp16>(), v.data_ptr<fp16>(), w.data_ptr<float>(), u.data_ptr<fp16>(), y.data_ptr<fp16>());
}
void forward_fp32_runner(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    cuda_forward_fp32_runner(B, T, C, H, r.data_ptr<fp32>(), k.data_ptr<fp32>(), v.data_ptr<fp32>(), w.data_ptr<float>(), u.data_ptr<fp32>(), y.data_ptr<fp32>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_bf16_runner", &forward_bf16_runner, "rwkv6 forward_bf16");
    m.def("forward_fp16_runner", &forward_fp16_runner, "rwkv6 forward_fp16");
    m.def("forward_fp32_runner", &forward_fp32_runner, "rwkv6 forward_fp32");
}
TORCH_LIBRARY(rwkv6, m) {
    m.def("forward_bf16_runner", forward_bf16_runner);
    m.def("forward_fp16_runner", forward_fp16_runner);
    m.def("forward_fp32_runner", forward_fp32_runner);
}
