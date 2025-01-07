

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <ATen/cuda/CUDAEvent.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>


__global__ void render_kernel(
    const int H, const int W, const int n,
    const int paramsM, const float *paramso_g, const float paramsgamma, const float *paramsnormal, const int paramsB,
    const float tan_psi,
    const float *K, const float *g_centers, const float *g_radius, const float *centers, const float *radius,
    const float n_air, const float n_water, uint8_t *rainimg, bool *mask, const uint8_t *refimg
) {
    const int x = blockIdx.x*blockDim.x+threadIdx.x;
	const int y = blockIdx.y*blockDim.y+threadIdx.y;
    
    if(y < H && x < W) {
        // to glass func
        float w = paramsM*tan_psi / (tan_psi - (y - K[1*3 + 2]) / K[1*3 + 1]);
        float u = w * (x - K[0*3 + 2]) / K[0*3 + 0];
        float v = w * (y - K[1*3 + 2]) / K[1*3 + 1];

        // in sphere raindrop func
        int sdist = 999;
        int idx = -1;
        for(int i = 0; i < n; i++) {
            float pu = u - g_centers[i*3 + 0];
            float pv = v - g_centers[i*3 + 1];
            float pw = w - g_centers[i*3 + 2];
            float norm_dist = sqrt(pu*pu + pv*pv + pw*pw);

            if (norm_dist <= g_radius[i]) {
                if (norm_dist < sdist) {
                    idx = i;
                    sdist = norm_dist;
                }
            }
        }

        // to sphere section env func
        if(idx != -1) {
            float alpha = acos((u*paramsnormal[0] + v*paramsnormal[1] + w*paramsnormal[2]) / sqrt(u*u + v*v + w*w));
            float beta = asin(n_air*sin(alpha) / n_water);

            float po_u = u - paramso_g[0];
            float po_v = v - paramso_g[1];
            float po_w = w - paramso_g[2];
            float po_norm = sqrt(po_u*po_u + po_v*po_v + po_w*po_w);
            po_u = po_u / po_norm;
            po_v = po_v / po_norm;
            po_w = po_w / po_norm;

            float tan_beta = tan(beta);
            float i_1_u = paramsnormal[0] + tan_beta*po_u;
            float i_1_v = paramsnormal[1] + tan_beta*po_v;
            float i_1_w = paramsnormal[2] + tan_beta*po_w;
            float i_1_norm = sqrt(i_1_u*i_1_u + i_1_v*i_1_v + i_1_w*i_1_w);
            i_1_u = i_1_u / i_1_norm;
            i_1_v = i_1_v / i_1_norm;
            i_1_w = i_1_w / i_1_norm;

            float oc_u = u - centers[idx*3 + 0];
            float oc_v = v - centers[idx*3 + 1];
            float oc_w = w - centers[idx*3 + 2];
            float tmp = i_1_u*oc_u + i_1_v*oc_v + i_1_w*oc_w;
            float d = -(tmp) + sqrt(pow(tmp, 2.0) - 1 + pow(radius[idx], 2.0));

            float p_w_u = u + d*i_1_u;
            float p_w_v = v + d*i_1_v;
            float p_w_w = w + d*i_1_w;

            float normal_w_u = p_w_u - centers[idx*3 + 0];
            float normal_w_v = p_w_v - centers[idx*3 + 1];
            float normal_w_w = p_w_w - centers[idx*3 + 2];
            float normal_w_norm = sqrt(normal_w_u*normal_w_u + normal_w_v*normal_w_v + normal_w_w*normal_w_w);
            normal_w_u = normal_w_u / normal_w_norm;
            normal_w_v = normal_w_v / normal_w_norm;
            normal_w_w = normal_w_w / normal_w_norm;

            d = ((p_w_u*normal_w_u + p_w_v*normal_w_v + p_w_w*normal_w_w) - (u*normal_w_u + v*normal_w_v + w*normal_w_w)) / (normal_w_u*normal_w_u + normal_w_v*normal_w_v + normal_w_w*normal_w_w);
            float p_a_u = p_w_u - (d*normal_w_u + u);
            float p_a_v = p_w_v - (d*normal_w_v + v);
            float p_a_w = p_w_w - (d*normal_w_w + w);
            float p_a_norm = sqrt(p_a_u*p_a_u + p_a_v*p_a_v + p_a_w*p_a_w);
            p_a_u = p_a_u / p_a_norm;
            p_a_v = p_a_v / p_a_norm;
            p_a_w = p_a_w / p_a_norm;

            float mx_u = p_w_u - u;
            float mx_v = p_w_v - v;
            float mx_w = p_w_w - w;
            float eta = acos((normal_w_u*mx_u + normal_w_v*mx_v + normal_w_w*mx_w) / sqrt(mx_u*mx_u + mx_v*mx_v + mx_w*mx_w));

            if(eta >= paramsgamma) {
                rainimg[y*W*3 + x*3 + 0] = 0;
                rainimg[y*W*3 + x*3 + 1] = 0;
                rainimg[y*W*3 + x*3 + 2] = 0;
                mask[y*W + x] = true;
            }
            else {
                float theta = tan(asin(n_water*sin(eta) / n_air));

                float i_2_u = normal_w_u + theta*p_a_u;
                float i_2_v = normal_w_v + theta*p_a_v;
                float i_2_w = normal_w_w + theta*p_a_w;

                float p_e_u = p_w_u + (paramsB - p_w_w) / i_2_w * i_2_u;
                float p_e_v = p_w_v + (paramsB - p_w_w) / i_2_w * i_2_v;
                float p_e_w = p_w_w + (paramsB - p_w_w) / i_2_w * i_2_w;

                int u_ = (int)round((K[0] * p_e_u + K[1] * p_e_v + K[2] * p_e_w) / paramsB);
                int v_ = (int)round((K[3] * p_e_u + K[4] * p_e_v + K[5] * p_e_w) / paramsB);

                if(u_ >= W) {
                    u_ = W - 1;
                }
                else if(u_ < 0) {
                    u_ = 0;
                }
                if(v_ >= H) {
                    v_ = H - 1;
                }
                else if(v_ < 0) {
                    v_ = 0;
                }

                rainimg[y*W*3 + x*3 + 0] = refimg[v_*W*3 + u_*3 + 0];
                rainimg[y*W*3 + x*3 + 1] = refimg[v_*W*3 + u_*3 + 1];
                rainimg[y*W*3 + x*3 + 2] = refimg[v_*W*3 + u_*3 + 2];
                mask[y*W + x] = true;
            }
        }
    }
}

void RainRenderCudaLauncher(
    const int H, const int W, const int n,
    const int paramsM, const float *paramso_g, const float paramsgamma, const float *paramsnormal, const int paramsB,
    const float tan_psi,
    const float *K, const float *g_centers, const float *g_radius, const float *centers, const float *radius,
    const float n_air, const float n_water, uint8_t *rainimg, bool *mask, const uint8_t *refimg
) {
    dim3 block_size;
	block_size.x = 32;
	block_size.y = 32;

	dim3 grid_size;
	grid_size.x = (W+block_size.x-1) / block_size.x;
	grid_size.y = (H+block_size.y-1) / block_size.y;

    render_kernel<<<grid_size, block_size>>>(
        H, W, n,
        paramsM, paramso_g, paramsgamma, paramsnormal, paramsB,
        tan_psi,
        K, g_centers, g_radius, centers, radius,
        n_air, n_water, rainimg, mask, refimg
    );

    //cudaDeviceSynchronize();
}