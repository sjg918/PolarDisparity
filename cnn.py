# refer : cgi-stereo
class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),)


    def forward(self, x):
        conv1 = self.conv1(x)

        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)

        conv3_up = self.conv3_up(conv3)

        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)

        conv2_up = self.conv2_up(conv2)

        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)

        conv = self.conv1_up(conv1)

        return conv


def context_upsample(depth_low, up_weights):
    ###
    # cv (b,1,h,w)
    # sp (b,9,4*h,4*w)
    ###
    b, c, h, w = depth_low.shape
        
    depth_unfold = F.unfold(depth_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    depth_unfold = F.interpolate(depth_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)

    depth = (depth_unfold*up_weights).sum(1)
        
    return depth


def regression_topk(cost, disparity_samples, k):

    _, ind = cost.sort(1, True)
    pool_ind = ind[:, :k]
    cost = torch.gather(cost, 1, pool_ind)
    prob = F.softmax(cost, 1)
    disparity_samples = torch.gather(disparity_samples, 1, pool_ind)    
    pred = torch.sum(disparity_samples * prob, dim=1, keepdim=True)
    return pred


class PolarPhaseGate(nn.Module):
    def __init__(self, alpha=10.0, eps=1e-6):
        super().__init__()
        self.alpha, self.eps = alpha, eps

    def forward(self, z: torch.Tensor):
        phi = torch.atan2(z.imag, z.real)
        score = torch.sin(phi)
        gate = torch.sigmoid(self.alpha * score).to(z.real.dtype)
        return torch.complex(gate * z.real, gate * z.imag)

# https://github.com/wavefrontshaping/complexPyTorch/blob/master/complexPyTorch/complexLayers.py
class _ComplexBatchNorm(nn.Module):

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 3))
            self.bias = nn.Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean_real", torch.zeros(
                    num_features, dtype=torch.float32)
            )
            self.register_buffer(
                "running_mean_imag", torch.zeros(
                    num_features, dtype=torch.float32)
            )
            self.register_buffer("running_covar", torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_covar", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean_real.zero_()
            self.running_mean_imag.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight[:, :2], 1.4142135623730951)
            nn.init.zeros_(self.weight[:, 2])
            nn.init.zeros_(self.bias)


class ComplexBatchNorm2d(_ComplexBatchNorm):
    def forward(self, inp):
        exponential_average_factor = 0.0
        running_mean = torch.complex(self.running_mean_real,self.running_mean_imag)

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                        float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = inp.real.mean([0, 2, 3]).type(torch.complex64)
            mean_i = inp.imag.mean([0, 2, 3]).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                running_mean = (
                    exponential_average_factor * mean
                    + (1 - exponential_average_factor) * running_mean
                )
                self.running_mean_real = running_mean.real
                self.running_mean_imag = running_mean.imag

        inp = inp - mean[None, :, None, None]

        if self.training or (not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = inp.numel() / inp.size(1)
            Crr = 1.0 / n * inp.real.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1.0 / n * inp.imag.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (inp.real.mul(inp.imag)).mean(dim=[0, 2, 3])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = (
                    exponential_average_factor * Crr * n / (n - 1)  #
                    + (1 - exponential_average_factor) * \
                    self.running_covar[:, 0]
                )

                self.running_covar[:, 1] = (
                    exponential_average_factor * Cii * n / (n - 1)
                    + (1 - exponential_average_factor) *
                    self.running_covar[:, 1]
                )

                self.running_covar[:, 2] = (
                    exponential_average_factor * Cri * n / (n - 1)
                    + (1 - exponential_average_factor) *
                    self.running_covar[:, 2]
                )

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        inp = (
            Rrr[None, :, None, None] * inp.real +
            Rri[None, :, None, None] * inp.imag
        ).type(torch.complex64) + 1j * (
            Rii[None, :, None, None] * inp.imag +
            Rri[None, :, None, None] * inp.real
        ).type(
            torch.complex64
        )

        if self.affine:
            inp = (
                self.weight[None, :, 0, None, None] * inp.real
                + self.weight[None, :, 2, None, None] * inp.imag
                + self.bias[None, :, 0, None, None]
            ).type(torch.complex64) + 1j * (
                self.weight[None, :, 2, None, None] * inp.real
                + self.weight[None, :, 1, None, None] * inp.imag
                + self.bias[None, :, 1, None, None]
            ).type(
                torch.complex64
            )
        return inp


class ComplexConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, bias=False):
        super().__init__()
        self.rw = nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias)
        self.iw = nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias)
    def forward(self, z):  # z: complex
        xr, xi = z.real, z.imag
        yr = self.rw(xr) - self.iw(xi)
        yi = self.rw(xi) + self.iw(xr)
        return torch.complex(yr, yi)


def polar_convbnact(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(ComplexConv2d(in_channels, out_channels, k=kernel_size, s=stride, p=padding, bias=False),
                         ComplexBatchNorm2d(out_channels),
                         PolarPhaseGate())

class SM(nn.Module):
    def __init__(self, inputH, inputW, complex=False):
        super().__init__()
        self.inl3d = INL3d(in_channels=4)
        self.cost_fusion = CostFusion(out_channels=8)
        self.hg = hourglass(in_channels=8)
        self.complex = complex
        self.top_k = 2
        self.maxdisp = 192
        self.disp_scale = 4
        self.ph_min = 0.
        self.ph_max = torch.pi
        self.ph_scale = (self.ph_max - self.ph_min) / self.maxdisp

        if complex:
            self.conv1 = polar_convbnact(2, 16, 7, 1, 3)
            self.conv2 = polar_convbnact(16, 16, 1, 1, 0)
            self.conv3 = polar_convbnact(16, 16, 3, 1, 1)
            self.conv4 = ComplexConv2d(16, 2, k=1, s=1, p=0, bias=True)
            self.polar_gate = PolarPhaseGate()
        else:
            self.spx = nn.Sequential(
                Conv_Bn_Activation(3, 32, 3, 2, 1, 'mish'),
                Conv_Bn_Activation(32, 64, 3, 2, 1, 'mish'),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReflectionPad2d((1, 0, 1, 0)),
                nn.AvgPool2d(2, stride=1), nn.BatchNorm2d(32), nn.Mish(inplace=True),
                nn.ConvTranspose2d(32, 9, kernel_size=4, stride=2, padding=1)
            )

    def build_exp_absdiff_costvol(self, refimg_fea, targetimg_fea):
        maxdisp = self.maxdisp // self.disp_scale
        num_groups = 4
        B, C, H, W = refimg_fea.shape
        assert C % num_groups == 0, "Number of channels must be divisible by number of groups"
        channels_per_group = C // num_groups

        ref = refimg_fea.view(B, num_groups, channels_per_group, H, W)
        tgt = targetimg_fea.view(B, num_groups, channels_per_group, H, W)

        volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])

        for d in range(maxdisp):
            if d > 0:
                cost = torch.abs(ref[:, :, :, :, d:] - tgt[:, :, :, :, :-d])
                cost = cost.mean(dim=2)
                volume[:, :, d, :, d:] = torch.exp(-cost)
            else:
                cost = torch.abs(ref - tgt)
                cost = cost.mean(dim=2)
                volume[:, :, d, :, :] = torch.exp(-cost)

        return volume.contiguous()

    def norm_correlation(self, fea1, fea2):
        cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True)
        return cost

    def build_norm_correlation_volume(self, refimg_fea, targetimg_fea):
        maxdisp = self.maxdisp // self.disp_scale
        B, C, H, W = refimg_fea.shape
        volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
        for i in range(maxdisp):
            if i > 0:
                volume[:, :, i, :, i:] = self.norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
            else:
                volume[:, :, i, :, :] = self.norm_correlation(refimg_fea, targetimg_fea)
        volume = volume.contiguous()
        return volume
    
    def build_polar_feat(self, cost, disparity_samples):
        B, D, H, W = cost.shape
        
        _, ind = cost.sort(1, True)
        pool_ind = ind[:, :self.top_k]
        cost = torch.gather(cost, 1, pool_ind)
        prob = F.softmax(cost, 1)
        disparity_samples = torch.gather(disparity_samples, 1, pool_ind)
        phi = disparity_samples * self.ph_scale + self.ph_min
        return torch.polar(prob, phi)
        
    def convert_polar2disp(self, feat: torch.Tensor, temp: float = 1.0, eps: float = 1e-8):
        assert torch.is_complex(feat), "complex branch expects complex feature map"

        mag = torch.abs(feat)                      # [B,K,H,W]  (m_k ≥ 0)
        phi = torch.atan2(feat.imag, feat.real)    # (-π, π]

        in_range = (phi >= self.ph_min) & (phi <= self.ph_max)
        mag = mag * in_range.to(mag.dtype)
        phi = phi.clamp(min=self.ph_min, max=self.ph_max)

        p = torch.softmax(mag / max(temp, 1e-4), dim=1)                    # Σ_k p_k = 1

        d = (phi - self.ph_min) / (self.ph_scale + eps)   # [B,K,H,W]

        disp = (p * d).sum(dim=1, keepdim=True)          # [B,1,H,W]
        disp = disp.clamp_(min=0.0, max=float(self.maxdisp - 1))

        return disp

    def forward(self, pred_left, leftx2, rightx2, leftx2_, rightx2_, leftx0):
        B, C, H, W = pred_left.shape
        exp_volume = self.inl3d(self.build_exp_absdiff_costvol(leftx2, rightx2))
        corr_volume = self.build_norm_correlation_volume(leftx2_, rightx2_)
        cost = self.cost_fusion(torch.cat((corr_volume, exp_volume), dim=1), leftx2, leftx0)
        cost = self.hg(cost)

        if self.complex:
            cost = F.interpolate(cost, (self.maxdisp, H, W), mode='trilinear')
            disp_samples = torch.arange(0, self.maxdisp, dtype=cost.dtype, device=cost.device)
            disp_samples = disp_samples.view(1, self.maxdisp, 1, 1).repeat(cost.shape[0],1,cost.shape[3],cost.shape[4])
            polar = self.build_polar_feat(cost.squeeze(1), disp_samples)
            polar = self.conv1(polar)
            polar = self.conv2(polar)
            polar = self.conv3(polar)
            polar = self.polar_gate(self.conv4(polar))
            pred_up = self.convert_polar2disp(polar)
            return pred_up.squeeze(1)
        else:
            spx_pred = self.spx(pred_left)
            spx_pred = F.softmax(spx_pred, 1)

            disp_samples = torch.arange(0, self.maxdisp//4, dtype=cost.dtype, device=cost.device)
            disp_samples = disp_samples.view(1, self.maxdisp//4, 1, 1).repeat(cost.shape[0],1,cost.shape[3],cost.shape[4])
            pred = regression_topk(cost.squeeze(1), disp_samples, 2)
            pred_up = context_upsample(pred, spx_pred)
            return pred_up * 4
