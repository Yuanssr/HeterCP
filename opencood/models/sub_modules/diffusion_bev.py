import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 嵌入与基本模块 ----------

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor):
        device = timesteps.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * -(math.log(10000) / (half - 1))
        )
        args = timesteps.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # [N, dim]


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU()
    )


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.act = nn.SiLU()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        t_feat = self.time_mlp(t_emb)[..., None, None]
        h = h + t_feat
        h = self.conv2(self.act(h))
        h = h + self.skip(x)
        return self.act(h)


class UNet2D(nn.Module):
    """
    条件通过通道拼接 (x_t || cond)，预测 x0 。
    """
    def __init__(self, in_ch, base_ch=128, t_dim=256, out_ch=None):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim)
        )
        out_ch = out_ch if out_ch is not None else in_ch

        self.down1 = ResBlock(in_ch, base_ch, t_dim)
        self.down2 = ResBlock(base_ch, base_ch * 2, t_dim)
        self.down3 = ResBlock(base_ch * 2, base_ch * 4, t_dim)
        self.pool = nn.MaxPool2d(2)

        self.mid = ResBlock(base_ch * 4, base_ch * 4, t_dim)

        self.up3 = ResBlock(base_ch * 8, base_ch * 2, t_dim)
        self.up2 = ResBlock(base_ch * 4, base_ch, t_dim)
        self.up1 = ResBlock(base_ch * 2, base_ch, t_dim)

        self.out = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x, cond, t):
        # x, cond: [B, C, H, W]
        t_emb = self.time_emb(t)

        inp = torch.cat([x, cond], dim=1)
        d1 = self.down1(inp, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        d3 = self.down3(self.pool(d2), t_emb)

        m = self.mid(self.pool(d3), t_emb)

        u3 = F.interpolate(m, scale_factor=2, mode="bilinear", align_corners=False)
        u3 = self.up3(torch.cat([u3, d3], dim=1), t_emb)

        u2 = F.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, d2], dim=1), t_emb)

        u1 = F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, d1], dim=1), t_emb)

        return self.out(u1)  # 预测 x0_hat


# ---------- 主扩散模块（x0 parameterization + 祖先式采样） ----------

def linear_beta_schedule(timesteps, start=5e-3, end=5e-2, device="cpu"):
    return torch.linspace(start, end, timesteps, device=device)


class DiffusionBEV(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.steps = model_cfg.get("timesteps", 1000)
        beta_start = model_cfg.get("beta_start", 5e-3)
        beta_end = model_cfg.get("beta_end", 5e-2)
        betas = linear_beta_schedule(self.steps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", torch.cat([torch.tensor([1.0], device=betas.device), alphas_cumprod[:-1]]))
        self.register_buffer("sqrt_ab", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_1m_ab", torch.sqrt(1 - alphas_cumprod))
        self.register_buffer("log_one_minus_ab", torch.log(1 - alphas_cumprod))
        self.register_buffer("sqrt_recip_ab", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_ab", torch.sqrt(1.0 / alphas_cumprod - 1))

        x_ch = model_cfg["x_channels"]        # 目标特征通道
        cond_ch = model_cfg["cond_channels"]  # 条件通道
        base_ch = model_cfg.get("base_channels", 128)
        t_dim = model_cfg.get("time_embed_dim", 256)
        self.unet = UNet2D(in_ch=x_ch + cond_ch, base_ch=base_ch, t_dim=t_dim, out_ch=x_ch)

    # --- q(x_t | x_0) ---
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self.sqrt_ab[t][:, None, None, None] * x_start +
            self.sqrt_1m_ab[t][:, None, None, None] * noise
        )

    # --- 后验系数 ---
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self.sqrt_recip_ab[t][:, None, None, None] * x_t -
            self.sqrt_recipm1_ab[t][:, None, None, None] * noise
        )

    def q_posterior(self, x_start, x_t, t):
        coef1 = (
            self.betas[t][:, None, None, None] *
            torch.sqrt(self.alphas_cumprod_prev[t][:, None, None, None]) /
            (1 - self.alphas_cumprod[t][:, None, None, None])
        )
        coef2 = (
            (1 - self.alphas_cumprod_prev[t][:, None, None, None]) *
            torch.sqrt(self.alphas_cumprod[t][:, None, None, None]) /
            (1 - self.alphas_cumprod[t][:, None, None, None])
        )
        model_mean = coef1 * x_start + coef2 * x_t
        posterior_variance = (
            self.betas[t][:, None, None, None] *
            (1 - self.alphas_cumprod_prev[t][:, None, None, None]) /
            (1 - self.alphas_cumprod[t][:, None, None, None])
        )
        posterior_log_variance = torch.log(torch.clamp(posterior_variance, min=1e-20))
        return model_mean, posterior_variance, posterior_log_variance

    # --- p(x_{t-1} | x_t) ---
    def p_mean_variance(self, x, cond, t, clip_denoised=False):
        x_recon = self.unet(x, cond, t)  # x0_hat
        if clip_denoised:
            x_recon = x_recon.clamp(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, cond, t, clip_denoised=False):
        model_mean, _, model_log_variance, x_recon = self.p_mean_variance(x, cond, t, clip_denoised)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x_recon

    @torch.no_grad()
    def p_sample_loop(self, cond, shape):
        b = shape[0]
        device = cond.device
        x = torch.randn(shape, device=device)
        x_recon_last = None
        for t in reversed(range(0, self.steps)):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            x, x_recon_last = self.p_sample(x, cond, t_batch, clip_denoised=False)
        return x, x_recon_last

    # 训练：输出 loss 和最终重建 x0_hat
    def forward(self, x0, cond):
        """
        x0: [N_non_ego_agents, C, H, W] 对应ego特征
        cond: [N_non_ego_agents, C, H, W] 条件特征
        """
        device = x0.device
        b = x0.shape[0]
        # 从 x0 加噪到 t=T-1，然后整链反推
        t_start = torch.full((b,), self.steps - 1, device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t_start, noise)
        # 反向链
        x_pred, x_recon_last = self.p_sample_loop(cond, x_t.shape)
        loss = F.mse_loss(x_pred, x0)
        # x_recon_last 是最后一步的 x0_hat，可用于监控
        return loss, x_recon_last

    # 推理采样：整链祖先式
    @torch.no_grad()
    def sample(self, cond, steps=None):
        # 若 steps 提供且 < self.steps，可子采样时间表
        if steps is None or steps >= self.steps:
            x, _ = self.p_sample_loop(cond, (cond.shape[0], self.unet.out.out_channels, cond.shape[2], cond.shape[3]))
            return x
        # 子采样时间表
        device = cond.device
        b, _, h, w = cond.shape
        x = torch.randn(b, self.unet.out.out_channels, h, w, device=device)
        idx = torch.linspace(self.steps - 1, 0, steps, device=device).long()
        for i, t_now in enumerate(idx):
            t_batch = t_now.expand(b)
            model_mean, _, model_log_variance, x_recon = self.p_mean_variance(x, cond, t_batch, clip_denoised=False)
            noise = torch.randn_like(x)
            nonzero_mask = (t_now != 0).float().reshape(b, *((1,) * (len(x.shape) - 1)))
            x = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x