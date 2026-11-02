"""Goal-conditioned diffusion policy: a 1D U-Net that denoises a whole pen path.

The task is generative, not discriminative: one goal raster admits many orderings and
many spacings of the same drawing, so a regression head would be trained toward the
mean of them all and would draw a blur. A conditional diffusion model learns the
distribution instead, and can be asked for as many distinct paths as a film needs.

Everything is in canvas units. The path is (K,3) - x, y, pen - with the pen carried at
+-1 so all three channels share one scale, which is what lets a single noise schedule
serve the geometry and the switch at the same time.
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import GOAL, K, pen_to_sign, sign_to_pen


class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        self.register_buffer("freq", torch.exp(
            -math.log(10000.0) * torch.arange(half).float() / max(half - 1, 1)))
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, t):
        a = t[:, None].float() * self.freq[None]
        return self.mlp(torch.cat([torch.sin(a), torch.cos(a)], dim=1))


class ResBlock(nn.Module):
    """Temporal conv block, conditioned on (diffusion step + goal) through FiLM."""

    def __init__(self, cin, cout, cond):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.film = nn.Linear(cond, cout * 2)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, c):
        h = self.c1(F.silu(self.n1(x)))
        g = self.film(c)[:, :, None]
        scale, shift = g[:, :h.shape[1]], g[:, h.shape[1]:]
        h = F.silu(self.n2(h)) * (1.0 + scale) + shift
        return self.skip(x) + self.c2(h)


class Attn(nn.Module):
    """One attention window over the whole path: a stroke's start has to be able to
    see its own end, which no 3-tap kernel can do at any depth."""

    def __init__(self, ch):
        super().__init__()
        self.n = nn.GroupNorm(8, ch)
        self.q = nn.Conv1d(ch, ch // 2, 1)
        self.k = nn.Conv1d(ch, ch // 2, 1)
        self.v = nn.Conv1d(ch, ch, 1)
        self.o = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        c, n = x.shape[1], x.shape[2]
        h = self.n(x)
        w = torch.softmax(self.q(h).transpose(1, 2) @ self.k(h) / math.sqrt(c // 2), dim=-1)
        return x + self.o((w @ self.v(h).transpose(1, 2)).transpose(1, 2))


class GoalEncoder(nn.Module):
    """The 48^2 ink picture is the only command the policy gets."""

    def __init__(self, dim, side=GOAL):
        super().__init__()
        def blk(ci, co, s):
            return nn.Sequential(nn.Conv2d(ci, co, 3, s, 1), nn.GroupNorm(8, co), nn.SiLU())
        self.net = nn.Sequential(blk(1, 32, 2), blk(32, 64, 2), blk(64, 128, 2),
                                 blk(128, 128, 2))
        self.proj = nn.Linear(128 * (side // 16) ** 2, dim)

    def forward(self, goal):
        return self.proj(self.net(goal[:, None]).flatten(1))


class UNet1D(nn.Module):
    """(K,3) noisy path + step + goal -> (K,3) predicted noise."""

    def __init__(self, k=K, cond=256, levels=(96, 128, 192)):
        super().__init__()
        assert len(levels) == 3, "the up path is written for three resolutions"
        # named on the module so the self-test can report the shape of the network it
        # actually built, rather than the shape somebody remembers passing as defaults
        self.levels, self.cond = tuple(levels), int(cond)
        self.t = TimeEmbed(cond // 2)
        self.g = GoalEncoder(cond // 2)
        cd = cond
        self.inp = nn.Conv1d(3, levels[0], 5, padding=2)
        self.down = nn.ModuleList([
            nn.ModuleList([ResBlock(levels[i - 1] if i else levels[0], lc, cd),
                           Attn(lc), ResBlock(lc, lc, cd)])
            for i, lc in enumerate(levels)])
        self.mid = nn.ModuleList([ResBlock(levels[-1], levels[-1], cd),
                                  Attn(levels[-1]), ResBlock(levels[-1], levels[-1], cd)])
        pairs = [(levels[2] * 2, levels[1]), (levels[1] * 2, levels[0]),
                 (levels[0] * 2, levels[0])]
        self.up = nn.ModuleList([
            nn.ModuleList([ResBlock(ci, co, cd), Attn(co), ResBlock(co, co, cd)])
            for ci, co in pairs])
        self.out = nn.Sequential(nn.GroupNorm(8, levels[0]), nn.SiLU(),
                                 nn.Conv1d(levels[0], 3, 5, padding=2))

    def forward(self, x, t, goal):
        c = torch.cat([self.t(t), self.g(goal)], dim=1)
        h = self.inp(x.transpose(1, 2))
        skips = []
        for rb, at, rb2 in self.down:
            h = rb2(at(rb(h, c)), c)
            skips.append(h)
            h = F.avg_pool1d(h, 2)
        h = self.mid[2](self.mid[1](self.mid[0](h, c)), c)
        for (rb, at, rb2), s in zip(self.up, reversed(skips)):
            h = F.interpolate(h, size=s.shape[-1], mode="linear", align_corners=False)
            h = rb2(at(rb(torch.cat([h, s], dim=1), c)), c)
        return self.out(h).transpose(1, 2)


def cosine_schedule(steps, clip=0.02):
    """Cosine alpha-bar, clipped so the last step is not asked to denoise pure signal.

    The argument to the cosine runs the full quarter turn from 0 to pi/2 as t goes 1 to
    `steps`: alpha-bar has to reach ~0 at the noisy end or the sampler starts from a
    distribution the model was never trained on.
    """
    u = torch.linspace(0.0, 1.0, steps + 1)
    ac = torch.cos((u + clip) / (1 + clip) * math.pi * 0.5) ** 2
    return (ac / ac[0])[1:].clamp(1e-5, 1.0)


class DiffusionPath:
    """DDPM training objective and ancestral sampler around UNet1D.

    Deliberately not an nn.Module: the schedule is fixed maths, not learned state, and
    keeping it outside means a checkpoint holds only what was trained.
    """

    def __init__(self, model, steps=100, device="cpu"):
        self.model, self.steps = model, int(steps)
        self.device = device
        ab = cosine_schedule(steps).to(device)
        self.ab = ab
        self.prev = torch.cat([ab.new_ones(1), ab[:-1]])
        # beta = 1 - ab_t/ab_{t-1}, clipped the way every DDPM implementation clips it:
        # the cosine schedule ends at alpha-bar ~0, and an unclipped 1/sqrt(alpha) is a
        # 6x amplifier on whatever the network has not learned to predict yet
        self.alpha = torch.clamp(ab / self.prev, min=1e-3)
        self.sigma = torch.clamp_min(
            (1.0 - self.prev) * (1.0 - ab / self.prev) / (1.0 - ab), 1e-20).sqrt()

    def loss(self, goal, path):
        b = path.shape[0]
        t = torch.randint(0, self.steps, (b,), device=path.device)
        ab = self.ab[t][:, None, None]
        eps = torch.randn_like(path)
        pred = self.model(path * ab.sqrt() + eps * (1.0 - ab).sqrt(), t.float(), goal)
        return F.mse_loss(pred, eps)

    @torch.no_grad()
    def sample(self, goal, k=K, seed=None, trace=False):
        """Ancestral DDPM sample. With trace, also the (steps+1,b,k,3) denoising film,
        starting at the pure noise the sampler began from."""
        b = goal.shape[0]
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(int(seed))
        x = torch.randn(b, k, 3, device=self.device, generator=g)
        states = [x.clone()] if trace else []
        for i in reversed(range(self.steps)):
            t = torch.full((b,), i, device=self.device)
            eps = self.model(x, t.float(), goal)
            ab, a, s = self.ab[i], self.alpha[i], self.sigma[i]
            mean = (x - (1.0 - ab).sqrt() * eps) / a.sqrt()
            x = mean if i == 0 else mean + s * torch.randn(
                x.shape, device=self.device, generator=g)
            if trace:
                states.append(x.clone())
        return (x, torch.stack(states)) if trace else x


SHIP = "policy_ship.pt"     # written by ckpt_select.py; the films read it by default


def load_policy(path, device):
    """(DiffusionPath, checkpoint dict) read from one saved file.

    One loader for the whole project, so the checkpoint that is measured, the one that is
    filmed and the one that is shipped cannot quietly be three different sets of weights.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    net = UNet1D().to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    pol = DiffusionPath(net, steps=int(ck["args"].get("steps", 100)), device=device)
    ck["ckpt_file"] = os.path.basename(path)
    pol.ckpt_file = ck["ckpt_file"]     # so a report can name the weights it measured
    return pol, ck


if __name__ == "__main__":
    torch.manual_seed(0)
    m = UNet1D()
    d = DiffusionPath(m, steps=20)
    goal = torch.rand(2, GOAL, GOAL)
    path = torch.randn(2, K, 3)
    n = sum(p.numel() for p in m.parameters())
    out = m(path, torch.zeros(2), goal)
    assert out.shape == path.shape, "model changed the path shape: %s" % (tuple(out.shape),)
    l = d.loss(goal, path)
    assert torch.isfinite(l), "loss is not finite"
    assert float(d.ab[0]) > 0.95 and float(d.ab[-1]) < 0.01, (
        "the schedule never reaches pure noise: alpha-bar %.4f to %.4f"
        % (float(d.ab[0]), float(d.ab[-1])))
    s = d.sample(goal[:1], seed=7)
    assert s.shape == (1, K, 3) and torch.isfinite(s).all(), "sampler produced junk"
    # the sampler has to depend on the goal, or the policy is an unconditional prior
    s2 = d.sample(goal[:1] * 0 + 1.0, seed=7)
    gap = float((s - s2).abs().mean())
    assert gap > 1e-3, "the goal changes nothing: same seed, mean |d path| %.5f" % gap
    # the denoising film is made from `trace`, so it has to hand back exactly the sample
    # it would have returned anyway, in noise-to-path order, and it has to actually move
    _, tr = d.sample(goal[:1], seed=7, trace=True)
    assert tr.shape == (d.steps + 1, 1, K, 3), "trace shape %s" % (tuple(tr.shape),)
    assert torch.equal(s, tr[-1]), "the last traced state differs from the sample"
    assert float((tr[0] - tr[-1]).abs().mean()) > 0.05, "the traced reverse process never moved"
    assert float(tr[0].std()) > 0.5, "state 0 of the trace is not the noise it started from"
    # the pen channel is stored as +-1 and the planner wants 0/1, so the two maps have
    # to be exact inverses over the range they are used on - and this is checked before
    # the PASS line is printed, not after it, or a failure here still looks like a pass.
    # a grid, not `path`: the tensor above is synthetic noise, and a round trip through
    # a clip is only an identity for a pen that was ever 0 or 1 in the first place.
    pen = np.linspace(0.0, 1.0, 257, dtype=np.float32)[:, None]
    rt = float(np.abs(sign_to_pen(pen_to_sign(pen)) - pen).max())
    assert rt < 1e-6, "the pen sign round trip drifts by %.6f" % rt
    print("MODEL PASS  params %.2f M  in %s -> out %s  loss %.4f"
          % (n / 1e6, tuple(path.shape), tuple(out.shape), float(l.detach())))
    print("        steps %d  alpha-bar[0] %.4f [-1] %.4f  goal-conditioning shift %.4f"
          % (d.steps, float(d.ab[0]), float(d.ab[-1]), gap))
    print("        levels %s  conditioning %d  pen sign round trip %.6f"
          % (" ".join(str(c) for c in m.levels), m.cond, rt))
