import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init

torch.set_default_dtype(torch.float64)


def ortho_estimate(mdl):
    l2_reg = None
    for W in mdl.parameters():
        if W.ndimension() < 2 or W.shape[1] == 1:
            continue
        else:
            cols = W[0].numel()
            rows = W.shape[0]
            w1 = W.view(-1, cols)
            wt = torch.transpose(w1, 0, 1)
            if (rows > cols):
                m = torch.matmul(wt, w1)
                ident = Variable(torch.eye(cols, cols), requires_grad=True)
            else:
                m = torch.matmul(w1, wt)
                ident = Variable(torch.eye(rows, rows), requires_grad=True)

            ident = ident.cuda()
            w_tmp = (m - ident)
            b_k = Variable(torch.rand(w_tmp.shape[1], 1))
            b_k = b_k.cuda()

            v1 = torch.matmul(w_tmp, b_k)
            norm1 = torch.norm(v1, 2)
            v2 = torch.div(v1, norm1)
            v3 = torch.matmul(w_tmp, v2)

            if l2_reg is None:
                l2_reg = (torch.norm(v3, 2)) ** 2
            else:
                l2_reg = l2_reg + (torch.norm(v3, 2)) ** 2
    return l2_reg

# ConvTranspose2d
def conv2d_block(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, stride=2, padding=2, *args, **kwargs),
        nn.Conv2d(in_f, out_f, stride=1, padding='same', *args, **kwargs),
        nn.PReLU(num_parameters=out_f, init=0.2)
    )


def convTr2d_block(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
        nn.ConvTranspose2d(in_f, out_f, stride=2, padding=2, output_padding=1, *args, **kwargs),
        nn.Conv2d(in_f, out_f, stride=1, padding='same', *args, **kwargs),
        # used to keep values positive, not significant coefficients selection.
        nn.PReLU(num_parameters=out_f, init=0.2)
    )




# def conv3d_block(in_f, out_f, *args, **kwargs):
#     return nn.Sequential(
#         nn.Conv3d(in_f, out_f, *args, **kwargs),
#         nn.Conv3d(in_f, out_f, *args, **kwargs),
#         nn.PReLU(num_parameters=out_f, init=0.2)
#     )

def orthogonal_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.orthogonal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)


def _init_weights(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=1.0)
        if module.bias is not None:
            module.bias.data.zero_()

class DSB_NET(nn.Module):
    def __init__(self, noise_sig, ker_size=5):
        super(DSB_NET, self).__init__()
        features = ker_size ** 2
        # self.wave_level = 3
        self.fd_sizes = [features, features]
        self.inv_sizes = [features, features]

        self.relu = nn.ReLU(inplace=True)
        self.prelu = nn.PReLU(num_parameters=features, init=0.2)
        self.prelus = nn.PReLU(num_parameters=1, init=0.2)
        # gradient
        self.lam = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True)
        self.gama = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True)
        self.mmu = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True)

        self.lam.data = 1 / (1e-2 * self.lam.data)
        self.gama.data = 1 / (3 * noise_sig * self.gama.data)
        self.mmu.data = 40 * self.mmu.data

        self.conv2_wt_up = nn.Conv2d(1, features, kernel_size=ker_size, stride=1, padding='same', bias=True,
                                     dtype=torch.float64)

        fd_2d_blocks = [
            conv2d_block(in_f, out_f, kernel_size=ker_size, dtype=torch.float64)
            for in_f, out_f in zip(self.fd_sizes, self.fd_sizes[1:])]

        self.trans_fd_l0 = nn.Sequential(*fd_2d_blocks)
        self.trans_fd_l1 = nn.Sequential(*fd_2d_blocks)
        self.trans_fd_l2 = nn.Sequential(*fd_2d_blocks)
        self.trans_fd_l3 = nn.Sequential(*fd_2d_blocks)

        bd_2d_blocks = [
            convTr2d_block(in_f, out_f, kernel_size=ker_size, dtype=torch.float64)
            for in_f, out_f in zip(self.fd_sizes, self.fd_sizes[1:])]

        self.trans_inv_l0 = nn.Sequential(*bd_2d_blocks)
        self.trans_inv_l1 = nn.Sequential(*bd_2d_blocks)
        self.trans_inv_l2 = nn.Sequential(*bd_2d_blocks)
        self.trans_inv_l3 = nn.Sequential(*bd_2d_blocks)

        self.conv2_back = nn.Conv2d(features, 1, kernel_size=ker_size, stride=1, padding='same', bias=True,
                                    dtype=torch.float64)

        # self.apply(_init_weights)
        self.apply(orthogonal_init)



    def Dx_f(self, u):
        d = torch.zeros_like(u)
        d[..., :, 2:] = u[..., :, 2:] - u[..., :, 1: - 1]
        d[..., :, 1] = u[..., :, 1] - u[..., :, -1]
        return d

    def Dxt_f(self, u):
        d = torch.zeros_like(u)
        d[..., :, 1: - 1] = u[..., :, 1: - 1] - u[..., :, 2:]
        d[..., :, -1] = u[..., :, -1] - u[..., :, 1]

        return d

    def Dy_f(self, u):
        d = torch.zeros_like(u)
        d[..., 2:, :] = u[..., 2:, :] - u[..., 1: - 1, :]
        d[..., 1, :] = u[..., 1, :] - u[..., -1, :]
        return d

    def Dyt_f(self, u):
        d = torch.zeros_like(u)
        d[..., 1: - 1, :] = u[..., 1: - 1, :] - u[..., 2:, :]
        d[..., -1, :] = u[..., -1, :] - u[..., 1, :]
        return d

    def wave_t(self, input_data):

        x = self.conv2_wt_up(input_data)
        x = self.prelu(x)
        x_0 = self.trans_fd_l0(x)
        x_1 = self.trans_fd_l1(x_0)
        x_2 = self.trans_fd_l2(x_1)
        x_3 = self.trans_fd_l3(x_2)

        return [x, x_0, x_1, x_2, x_3]

    def wave_inv(self, w_coef):
        x_iv_l3 = self.trans_inv_l1(w_coef[-1]) + w_coef[-2]
        x_iv_l2 = self.trans_inv_l1(x_iv_l3) + w_coef[-3]
        x_iv_l1 = self.trans_inv_l2(x_iv_l2) + w_coef[-4]
        x_iv_l0 = self.trans_inv_l3(x_iv_l1) + w_coef[0]

        rec_img = self.conv2_back(x_iv_l0)
        rec_img = self.prelus(rec_img)
        return rec_img

    def w_add(self, w_list, ww_list):
        x_h = []
        for i in range(len(w_list)):
            x_h.append(w_list[i] + ww_list[i])

        return x_h

    def w_sub(self, w_list, wa_list):
        x_h = []
        for i in range(len(w_list)):
            x_h.append(w_list[i] - wa_list[i])

        return x_h

    def set_func_grad(self, is_grad=True):
        self.conv2_wt_up.requires_grad_(requires_grad=is_grad)
        self.trans_fd_l0.requires_grad_(requires_grad=is_grad)
        self.trans_fd_l1.requires_grad_(requires_grad=is_grad)
        self.trans_fd_l2.requires_grad_(requires_grad=is_grad)
        self.trans_fd_l3.requires_grad_(requires_grad=is_grad)

        self.trans_inv_l0.requires_grad_(requires_grad=is_grad)
        self.trans_inv_l1.requires_grad_(requires_grad=is_grad)
        self.trans_inv_l2.requires_grad_(requires_grad=is_grad)
        self.trans_inv_l3.requires_grad_(requires_grad=is_grad)
        self.conv2_back.requires_grad_(requires_grad=is_grad)


    def forward(self, u, uvMask, dirtyMap, f, bg_list, is_para=False):

        self.set_func_grad(is_grad=is_para)

        w = self.wave_t(u)

        w_t = self.wave_t(dirtyMap)

        img = self.wave_inv(w_t)

        sym_loss = img - dirtyMap

        bx = bg_list[0]
        by = bg_list[1]
        bw = bg_list[2]
        dx = bg_list[3]
        dy = bg_list[4]

        # dx = torch.zeros_like(dirtyMap)
        # dy = torch.zeros_like(dirtyMap)
        #
        # bx = torch.zeros_like(dirtyMap)
        # by = torch.zeros_like(dirtyMap)
        # bw = self.wave_t(torch.zeros_like(dirtyMap))

        f0 = f

        murf = torch.fft.ifft2(torch.multiply(uvMask, f)) * self.mmu


        uker = torch.zeros_like(dirtyMap)
        uker[..., 1, 1] = 4
        uker[..., 1, 2] = -1
        uker[..., 2, 1] = -1
        uker[..., -1, 1] = -1
        uker[..., 1, -1] = -1

        uker = (torch.multiply(torch.conj(uvMask), uvMask) * self.mmu) + self.lam * torch.fft.fft2(uker) + self.gama
        # first update image
        for _ in range(2):

            rhs = murf + self.lam * self.Dxt_f(torch.sub(dx, bx)) + self.lam * self.Dyt_f(
                torch.sub(dy, by)) + self.gama * self.wave_inv(
                self.w_sub(w, bw))


            u = torch.real(torch.fft.ifft2(torch.divide(torch.fft.fft2(rhs), uker)))

            u = self.prelus(u)
            s_k = torch.sqrt(torch.square(dx + bx) + torch.square(dy + by))

            dx = self.relu((dx + bx) - (1 / self.lam) * s_k)
            dy = self.relu((dy + by) - (1 / self.lam) * s_k)

            w_ce = self.w_add(self.wave_t(u), bw)
            w = []
            for i in range(len(w_ce)):
                w.append(self.relu(w_ce[i]-(1 / self.gama)))

            bx = bx + self.Dx_f(u) - dx
            by = by + self.Dy_f(u) - dy
            bw = self.w_add(bw, self.w_sub(self.wave_t(u), w))

            f = f + f0 - torch.multiply(uvMask, torch.fft.fft2(u))
            murf = torch.fft.ifft2(torch.multiply(uvMask, f)) * self.mmu


        # then update dict through backpropagation
        return u, sym_loss


if __name__ == '__main__':
    # the maximum number of iterations
    max_iterations = 1000
    # the number of base vectors in each level =  kernel_size x kernel_size
    kernel_size = 5

    # mse 3.5221144353452715e-05

    # optimization scheduler and learning rate
    key_step = 100
    lrate_img = 1e-1
    gamma_lr = 1e-2
    lrate_dict = 1e-3
    is_update = False




    simplenoise = 50
    # see https://casadocs.readthedocs.io/en/stable/api/tt/casatools.simulator.html#casatools.simulator.simulator.setnoise
    sigma_ = simplenoise / np.sqrt((27 * 26) / 2 * (28800 / 60))
    print('noise sigma',sigma_)

    PSF_mat = sio.loadmat('data/psf.mat')
    PSF = PSF_mat['PSF']

    skyModel_mat = sio.loadmat('data/skyModel.mat')
    skyModel = skyModel_mat['skyModel']

    Dirtymap_mat = sio.loadmat('data/Dirtymap.mat')
    Dirtymap = Dirtymap_mat['Dirtymap']

    fig, axs = plt.subplots(1, 3)
    p0 = axs[0].imshow(skyModel, cmap='turbo', origin='upper')
    p1 = axs[1].imshow(PSF, cmap='turbo', origin='upper')
    p2 = axs[2].imshow(Dirtymap, cmap='turbo')


    axs[0].set_title('skyModel')
    axs[1].set_title('PSF')
    axs[2].set_title('Dirtymap')

    plt.show()

    m = np.unravel_index(PSF.argmax(), PSF.shape)
    center = 1 - np.array(m)
    UV = np.fft.fft2(np.roll(PSF, center - 1, axis=(0, 1)))

    # convert numpy array to tensor and init values
    t_UV = torch.from_numpy(UV)
    t_UV = t_UV[None, None, :, :].cuda()
    t_dirtyMap = torch.from_numpy(Dirtymap)
    t_dirtyMap = t_dirtyMap[None, None, :, :].cuda()
    ft = torch.fft.fft2(t_dirtyMap).cuda()
    init_img = torch.zeros_like(t_dirtyMap).cuda()

    # Bregman method variables
    bw_l = torch.zeros(1, kernel_size**2, 256, 256).cuda()
    bw_l0 = torch.zeros(1, kernel_size**2, 128, 128).cuda()
    bw_l1 = torch.zeros(1, kernel_size**2, 64, 64).cuda()
    bw_l2 = torch.zeros(1, kernel_size**2, 32, 32).cuda()
    bw_l3 = torch.zeros(1, kernel_size**2, 16, 16).cuda()

    bw0 = [bw_l, bw_l0, bw_l1, bw_l2, bw_l3]
    bx0 = torch.zeros_like(t_dirtyMap).cuda()
    by0 = torch.zeros_like(t_dirtyMap).cuda()

    dx0 = torch.zeros_like(t_dirtyMap).cuda()
    dy0 = torch.zeros_like(t_dirtyMap).cuda()

    bg_list = [bx0, by0, bw0, dx0, dy0]

    # create deep learning model
    model = DSB_NET(noise_sig=sigma_, ker_size=kernel_size)
    # print('model', model)
    model.cuda()

    optimizer_image = torch.optim.Adam(model.parameters(), lr=lrate_img)
    scheduler_image = torch.optim.lr_scheduler.MultiStepLR(optimizer_image, milestones=[key_step], gamma=gamma_lr)

    mse_loss = nn.MSELoss()

    recons_img = torch.zeros(t_dirtyMap.shape)
    min_residual = torch.zeros(t_dirtyMap.shape)
    min_loss = 1e9

    for step in range(max_iterations):

        recons_temp, loss_layers_sym = model(init_img, t_UV, t_dirtyMap, ft, bg_list, is_update)

        tmp_dirty = torch.real((torch.fft.ifft2(torch.mul(t_UV, torch.fft.fft2(recons_temp)))))
        loss_mse = mse_loss(tmp_dirty, t_dirtyMap)

        loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
        gamma = torch.Tensor([1e-3]).cuda()

        or_lambda = torch.Tensor([1e-2]).cuda()
        or_loss = ortho_estimate(model).cuda()

        loss_all = loss_mse + torch.mul(gamma, loss_constraint.item()) + torch.mul(or_lambda, or_loss)

        optimizer_image.zero_grad()
        loss_all.backward()
        optimizer_image.step()
        scheduler_image.step()

        print('step {}, fidelity {},sym {}, ortho {}, all {}'.format(step, loss_mse.data.cpu(),

                                                                     torch.mul(gamma,
                                                                               loss_constraint.item()).cpu().item(),
                                                                     torch.mul(or_lambda, or_loss).cpu().item(),
                                                                     loss_all.cpu().item()
                                                                     ))

        if step == key_step:
            is_update = True

        if loss_mse.item() < min_loss:
            recons_img = recons_temp
            min_loss = loss_mse.item()
            o_residual = t_dirtyMap - tmp_dirty

        if step == max_iterations - 1:
            # print('save trained model')
            # torch.save(model.state_dict(), 'm31_weights.pth')

            recons_img = torch.squeeze(recons_img)
            recons_img = recons_img.cpu().detach().numpy()

            residual_img = torch.squeeze(o_residual)
            residual_img = residual_img.cpu().detach().numpy()

            self_mse = np.mean(np.square(skyModel - recons_img))
            print('mse', self_mse)

            dis_scale = 3.0
            fig, axs = plt.subplots(1, 3)
            p0 = axs[0].imshow(np.abs(skyModel - recons_img), cmap='turbo', origin='upper', vmin=0.0,
                               vmax=dis_scale * sigma_)
            p1 = axs[1].imshow(recons_img, cmap='turbo', origin='upper', vmin=-0.0, vmax=1.0)

            p2 = axs[2].imshow(residual_img, cmap='turbo', origin='upper', vmin=-dis_scale * sigma_, vmax=dis_scale * sigma_)

            fig.colorbar(p1)
            fig.colorbar(p0)
            fig.colorbar(p2)

            axs[0].set_title('abs error')
            axs[1].set_title('reconstruction')
            axs[2].set_title('residual')

            plt.show()
            break
