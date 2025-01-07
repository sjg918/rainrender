
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cuda.h>
#include <torch/extension.h>

void RainRenderCudaLauncher(
    const int H, const int W, const int n,
    const int paramsM, const float *paramso_g, const float paramsgamma, const float *paramsnormal, const int paramsB,
    const float tan_psi,
    const float *K, const float *g_centers, const float *g_radius, const float *centers, const float *radius,
    const float n_air, const float n_water, uint8_t *rainimg, bool *mask, const uint8_t *refimg
);

at::Tensor renderExtension(
    const int H, const int W, const int n,
    const int paramsM, const at::Tensor paramso_g, const float paramsgamma, const at::Tensor paramsnormal, const int paramsB,
    const float tan_psi,
    const at::Tensor K, const at::Tensor g_centers, const at::Tensor g_radius, const at::Tensor centers, const at::Tensor radius,
    const float n_air, const float n_water, const at::Tensor refimg, at::Tensor rainimg
) {

    float* paramso_g_ = paramso_g.contiguous().data_ptr<float>();
    float* paramsnormal_ = paramsnormal.contiguous().data_ptr<float>();
    float* K_ = K.contiguous().data_ptr<float>();
    float* g_centers_ = g_centers.contiguous().data_ptr<float>();
    float* g_radius_ = g_radius.contiguous().data_ptr<float>();
    float* centers_ = centers.contiguous().data_ptr<float>();
    float* radius_ = radius.contiguous().data_ptr<float>();
    const uint8_t * refimg_ = refimg.contiguous().data_ptr<uint8_t>();
    uint8_t* rainimg_ = rainimg.contiguous().data_ptr<uint8_t>();

    auto mask = at::zeros({H, W}, torch::dtype(torch::kBool)).to(refimg.device());
    bool* mask_ = mask.data_ptr<bool>();

    RainRenderCudaLauncher(
        H, W, n,
        paramsM, paramso_g_, paramsgamma, paramsnormal_, paramsB,
        tan_psi,
        K_, g_centers_, g_radius_, centers_, radius_,
        n_air, n_water, rainimg_, mask_, refimg_
    );

    return mask;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("renderExtension", &renderExtension,
        "rain render cuda");
}