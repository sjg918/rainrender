
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import cv2
import os
import math
import torch
import rr_cuda

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def to_glass(x, y, M, tan_psi, K):
    w = M * tan_psi / (tan_psi - (y - K[1, 2]) / K[1, 1]);
    u = w * (x - K[0, 2]) / K[0, 0]
    v = w * (y - K[1, 2]) / K[1, 1]

    return np.array((u, v, w))

def w_in_plane(u, v, M, normal):
    return (normal[2]*M - normal[0]*u - normal[1]*v) / normal[2]

def get_sphere_raindrop(W, H, n, M, tan_psi, K, normal, baseline=0.54):
    g_centers = []
    g_radius = []
    centers = []
    radius = []

    left_upper = to_glass(0, 0, M, tan_psi, K)
    left_bottom = to_glass(0, H, M, tan_psi, K)
    # right_upper = to_glass(W, 0, M, tan_psi, K)
    right_bottom = to_glass(W, H, M, tan_psi, K)
    right_bottom[0] = right_bottom[0] + baseline * 1000

    for i in range(n):
        u = left_bottom[0] + (right_bottom[0] - left_bottom[0]) * np.random.uniform(0, 1)
        v = left_upper[1] + (right_bottom[1] - left_upper[1]) * np.random.uniform(0, 1)
        w = w_in_plane(u, v, M, normal)

        tau = np.random.randint(30, 45) / 180 * np.pi
        glass_r = 0.8 + 0.6  * np.random.uniform(2, 3.2)

        r_sphere = glass_r / math.sin(tau)

        g_c = np.array([u, v, w])
        c = g_c - normal * r_sphere * math.cos(tau)

        g_centers.append(g_c)
        g_radius.append(glass_r)
        centers.append(c)
        radius.append(r_sphere)
        continue
    
    g_centers = np.array(g_centers)
    g_radius = np.array(g_radius)
    centers = np.array(centers)
    radius = np.array(radius)

    return g_centers, g_radius, centers, radius

if __name__ == '__main__':
    files = os.listdir('./Limg/')
    n_air = 1.0
    n_water = 1.33
    baseline = 0.54
    paramsgamma = math.asin(1.00 / 1.33)
    
    with torch.no_grad():
        for imgname in files:
            Limg = cv2.imread('./Limg/' + imgname)
            Rimg = cv2.imread('./Rimg/' + imgname)
            Limg = Limg[:352, :1216, :]
            Rimg = Rimg[:352, :1216, :]
            H, W, _ = Limg.shape
            Lrainimg = Limg.copy()
            Rrainimg = Rimg.copy()

            filedata = read_calib_file('calib.txt')
            P_rect_02 = np.reshape(filedata['P2'], (3, 4))
            K_left = P_rect_02[0:3, 0:3]
            P_rect_03 = np.reshape(filedata['P3'], (3, 4))
            K_right = P_rect_03[0:3, 0:3]

            n = np.random.randint(100, 200)

            #paramsM = np.random.randint(100, 500)
            paramsM = 150
            paramsB = np.random.randint(4000, 8000)
            paramspsi = np.random.randint(30, 45)  / 180.0 * np.pi
            tan_psi = np.tan(paramspsi)

            paramsnormal = np.array((0.0, -1.0 * math.cos(paramspsi), math.sin(paramspsi)))
            paramso_g = (paramsnormal[2] * paramsM) * paramsnormal

            pack = get_sphere_raindrop(W, H, n, paramsM, tan_psi, K_left, paramsnormal, baseline=baseline) # must use K_left
            g_centers, g_radius, centers, radius = pack

            # n
            # paramsM
            paramso_g = torch.from_numpy(paramso_g).cuda().to(torch.float32)
            # paramsgamma
            paramsnormal = torch.from_numpy(paramsnormal).cuda().to(torch.float32)
            # paramsB
            tan_psi = tan_psi.item()
            K_left = torch.from_numpy(K_left).cuda().to(torch.float32)
            K_right = torch.from_numpy(K_right).cuda().to(torch.float32)
            g_centers = torch.from_numpy(g_centers).cuda().to(torch.float32)
            g_radius = torch.from_numpy(g_radius).cuda().to(torch.float32)
            centers = torch.from_numpy(centers).cuda().to(torch.float32)
            radius = torch.from_numpy(radius).cuda().to(torch.float32)
            Limg = torch.from_numpy(Limg).cuda()
            Lrainimg = torch.from_numpy(Lrainimg).cuda()
            Rimg = torch.from_numpy(Rimg).cuda()
            Rrainimg = torch.from_numpy(Rrainimg).cuda()

            Lmask = rr_cuda.renderExtension(
                H, W, n,
                paramsM, paramso_g, paramsgamma, paramsnormal, paramsB,
                tan_psi,
                K_left, g_centers, g_radius, centers, radius,
                n_air, n_water, Limg, Lrainimg)
            
            g_centers[:, 0] = g_centers[:, 0] - baseline * 1000
            centers[:, 0] = centers[:, 0] - baseline * 1000
            Rmask = rr_cuda.renderExtension(
                H, W, n,
                paramsM, paramso_g, paramsgamma, paramsnormal, paramsB,
                tan_psi,
                K_right, g_centers, g_radius, centers, radius,
                n_air, n_water, Rimg, Rrainimg)

            Lrainimg = Lrainimg.cpu().numpy()
            Lmask = Lmask.cpu().numpy()
            Rrainimg = Rrainimg.cpu().numpy()
            Rmask = Rmask.cpu().numpy()
            
            Lblurimg = cv2.GaussianBlur(Lrainimg, (0,0), np.random.uniform(1,1.5))
            Lmask = np.tile(Lmask.reshape(H, W, 1), (1,1,3))
            Lrainimg[Lmask] = Lblurimg[Lmask]
            cv2.imwrite('./out/L_' + imgname, Lrainimg)

            Rblurimg = cv2.GaussianBlur(Rrainimg, (0,0), np.random.uniform(1,1.5))
            Rmask = np.tile(Rmask.reshape(H, W, 1), (1,1,3))
            Rrainimg[Rmask] = Rblurimg[Rmask]
            cv2.imwrite('./out/R_' + imgname, Rrainimg)

            #df=df
            continue