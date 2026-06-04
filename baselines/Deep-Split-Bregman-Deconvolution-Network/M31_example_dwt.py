import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

wavelet_used = 'sym7'
from pytorch_wavelets import DWTForward, DWTInverse
wave_level = 1
torch.set_default_dtype(torch.float64)
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False
        wave_level = 1
        self.xfm = DWTForward(J=wave_level, mode='reflect', wave=wavelet_used).cuda()

    def forward(self, x):
        return self.dwt_init(x)

    def dwt_init(self, x):
        Yl, Yh = self.xfm(x)
        coeif_full=[]
        coeif_full.append(Yl)
        coeif_full.append(Yh)
        return coeif_full


class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False
        self.ifm = DWTInverse(mode='reflect', wave=wavelet_used).cuda()

    def forward(self, x):
        return self.idwt_init(x)

    def idwt_init(self, x):
        Y = self.ifm((x[0], x[1]))
        return Y

class DSB_NET(nn.Module):
    def __init__(self,n_sig):
        super(DSB_NET, self).__init__()
        self.wave_level = 1
        self.DWT = DWT().cuda()
        self.IDWT = IDWT().cuda()


        self.lam = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True)
        self.gama = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True)
        self.mmu = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True)


        self.lam.data = 1 / (3 * n_sig) * self.lam.data
        self.gama.data = 1 / (1 * n_sig) * self.gama.data
        self.mmu.data = 40 * self.mmu.data
        self.srelu = nn.PReLU(num_parameters=1, init=0.2)

    def Dx_f(self, u):
        d = torch.zeros_like(u)
        d[...,:, 2:] = u[...,:, 2:] - u[...,:, 1: - 1]
        d[...,:, 1] = u[...,:, 1] - u[...,:, -1]
        return d

    def Dxt_f(self, u):
        d = torch.zeros_like(u)

        d[..., :, 1: - 1] = u[..., :, 1: - 1] - u[..., :, 2:]
        d[..., :, -1] = u[..., :, -1] - u[..., :, 1]

        return d

    def Dy_f(self, u):
        d = torch.zeros_like(u)
        d[...,2:, :] = u[...,2:, :] - u[...,1: - 1, :]
        d[...,1, :] = u[...,1, :] - u[...,-1, :]
        return d

    def Dyt_f(self, u):
        d = torch.zeros_like(u)
        d[...,1: - 1, :] = u[...,1: - 1, :] - u[...,2:, :]
        d[...,-1, :] = u[...,-1, :] - u[...,1, :]
        return d

    def coef_add(self, c1, c2):
        Yl_c1 = c1[0]
        Yh_c1 = c1[1]

        Yl_c2 = c2[0]
        Yh_c2 = c2[1]

        out_coef = []
        out_coef.append(torch.add(Yl_c1 , Yl_c2))

        Yh_out = []
        for i in range(len(Yh_c1)):
            Yh_out.append(torch.add(Yh_c1[i], Yh_c2[i]))

        out_coef.append(Yh_out)
        return out_coef

    def coef_subs(self, c1, c2):
        Yl_c1 = c1[0]
        Yh_c1 = c1[1]

        Yl_c2 = c2[0]
        Yh_c2 = c2[1]

        out_coef = []
        out_coef.append(torch.subtract(Yl_c1, Yl_c2))

        Yh_out = []
        for i in range(len(Yh_c1)):
            Yh_out.append(torch.subtract(Yh_c1[i], Yh_c2[i]))

        out_coef.append(Yh_out)
        return out_coef

    def forward(self, u, uvMask, dirtyMap, f):
        w = self.DWT(u)

        dx = torch.zeros_like(dirtyMap)
        dy = torch.zeros_like(dirtyMap)

        bx = torch.zeros_like(t_dirtyMap)
        by = torch.zeros_like(t_dirtyMap)
        bw = self.DWT(torch.zeros_like(dirtyMap))

        f0 = f
        murf = torch.fft.ifft2(torch.multiply(uvMask, f))* self.mmu

        uker = torch.zeros_like(dirtyMap)
        uker[..., 1, 1] = 4
        uker[..., 1, 2] = -1
        uker[..., 2, 1] = -1
        uker[..., -1, 1] = -1
        uker[..., 1, -1] = -1

        uker = (torch.multiply(torch.conj(uvMask), uvMask) * self.mmu) + self.lam * torch.fft.fft2(uker) + self.gama


        for _ in range(2):

            rhs = murf + self.lam * self.Dxt_f(dx - bx) + self.lam * self.Dyt_f(
                dy - by) + self.gama * self.IDWT(self.coef_subs(w, bw))

            u = torch.real(torch.fft.ifft2(torch.divide(torch.fft.fft2(rhs), uker)))

            u = self.srelu(u)
            s_k = torch.sqrt(torch.square(dx + bx) + torch.square(dy + by))

            dx = self.srelu((dx + bx) - (1 / self.lam) * s_k)
            dy = self.srelu((dy + by) - (1 / self.lam) * s_k)

            w_ups1 = self.coef_add(self.DWT(u), bw)

            w_yl = self.srelu(w_ups1[0] - 1 / self.gama)
            w_yh = w_ups1[1]
            w_update_yh = []
            for i in range(len(w_yh)):
                w_update_yh.append(self.srelu(w_yh[i] - 1 / self.gama))

            w = [w_yl, w_update_yh]

            bx = bx + self.Dx_f(u) - dx
            by = by + self.Dy_f(u) - dy
            bw_s1u =self.coef_subs(self.DWT(u), w)
            bw = self.coef_add(bw, bw_s1u)

            f = f + f0 - torch.multiply(uvMask, torch.fft.fft2(u))
            murf = torch.fft.ifft2(torch.multiply(uvMask, f)) * self.mmu

        return u

if __name__ == '__main__':
    sttpps = 1000

    noise_sigma = 50
    sigma_ = noise_sigma / np.sqrt((27 * 26) / 2 * (28800 / 60))
    con_thrs = sigma_ ** 2

    PSF_mat = sio.loadmat('data/psf.mat')
    PSF = PSF_mat['PSF']

    skyModel_mat = sio.loadmat('data/skyModel.mat')
    skyModel = skyModel_mat['skyModel']

    Dirtymap_mat = sio.loadmat('data/Dirtymap.mat')
    Dirtymap = Dirtymap_mat['Dirtymap']

    m = np.unravel_index(PSF.argmax(), PSF.shape)
    center = 1 - np.array(m)
    UV = np.fft.fft2(np.roll(PSF, center - 1, axis=(0, 1)))

    t_UV = torch.from_numpy(UV)
    t_UV = t_UV[None, None, :, :].cuda()
    t_dirtyMap = torch.from_numpy(Dirtymap)
    t_dirtyMap = t_dirtyMap[None, None, :, :].cuda()

    ft = torch.fft.fft2(t_dirtyMap).cuda()
    recons_temp = torch.zeros_like(t_dirtyMap).cuda()
    init_img = torch.zeros_like(t_dirtyMap).cuda()


    model = DSB_NET(n_sig=sigma_)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    mse_loss = nn.MSELoss()
    recons_img = torch.zeros(t_dirtyMap.shape)
    min_residual = torch.zeros(t_dirtyMap.shape)
    min_loss = 1e9

    for step in range(sttpps):
        recons_temp = model(init_img, t_UV, t_dirtyMap, ft)
        tmp_dirty = torch.real((torch.fft.ifft2(torch.mul(t_UV, torch.fft.fft2(recons_temp)))))
        loss_mse = mse_loss(tmp_dirty, t_dirtyMap)

        optimizer.zero_grad()
        loss_mse.backward()
        optimizer.step()
        scheduler.step()

        print('step {}, fidelity {}'.format(step, loss_mse.data.cpu()))
        if loss_mse.item() < min_loss:
            recons_img = recons_temp
            min_loss = loss_mse.item()
            min_residual = t_dirtyMap - tmp_dirty

        if step == sttpps - 1:
            # print('save trained model')
            # torch.save(model.state_dict(), 'm31_weights.pth')

            recons_img = torch.squeeze(recons_img)
            recons_img = recons_img.cpu().detach().numpy()

            residual_img = torch.squeeze(min_residual)
            residual_img = residual_img.cpu().detach().numpy()

            self_mse = np.mean(np.square(skyModel - recons_img))
            print('mse', self_mse)

            dis_scale = 3.0
            fig, axs = plt.subplots(1, 3)
            p0 = axs[0].imshow(np.abs(skyModel - recons_img), cmap='turbo', origin='upper', vmin=0.0,
                               vmax=dis_scale * sigma_)
            p1 = axs[1].imshow(recons_img, cmap='turbo', origin='upper', vmin=-0.0, vmax=1.0)

            p2 = axs[2].imshow(residual_img, cmap='turbo', origin='upper', vmin=-dis_scale * sigma_,
                               vmax=dis_scale * sigma_)

            fig.colorbar(p1)
            fig.colorbar(p0)
            fig.colorbar(p2)

            axs[0].set_title('abs error')
            axs[1].set_title('reconstruction')
            axs[2].set_title('residual')

            plt.show()
            break



