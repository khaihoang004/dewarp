import copy
import time
import unittest

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

from src.model.loop_rep.model import LoopRepDocEnhanceNet, AdaptiveLoopedBottleneck


ATOL = 1e-4


def max_abs_error(a, b):
    return (a - b).abs().max().item()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def benchmark_latency(model, dummy_input, device, warmup=10, iters=50):
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(iters):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.time()

    return (end - start) * 1000 / iters


class TestLoopRepDocEnhanceNet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.base_dim = 32
        cls.max_loops = 4

        cls.model = LoopRepDocEnhanceNet(
            base_dim=cls.base_dim,
            max_loops=cls.max_loops,
            deploy=False
        ).to(cls.device)

        cls.input = torch.randn(1, 3, 256, 256, device=cls.device)

    # ====================== Training Mode ======================
    def test_training_forward(self):
        self.model.train()
        output, inter_preds, halting_weights, gate_logits, exit_probs = self.model(
            self.input, return_all=True
        )

        self.assertEqual(output.shape, (1, 3, 256, 256))
        self.assertEqual(len(inter_preds), self.max_loops)
        self.assertEqual(len(halting_weights), self.max_loops)

    def test_halting_weights_sum_to_one(self):
        self.model.train()
        _, _, halting_weights, _, _ = self.model(self.input, return_all=True)

        total = halting_weights.sum(dim=0)
        self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1e-5))

    # ====================== Inference Mode ======================
    def test_inference_forward(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.input)

        self.assertEqual(output.shape, (1, 3, 256, 256))

    def test_output_range(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.input)
        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output <= 1.0))

    # ====================== Fuse ======================
    def test_fuse_consistency(self):
        self.model.eval()
        with torch.no_grad():
            out_before = self.model(self.input)

        fused = copy.deepcopy(self.model)
        fused.fuse_entire_model()

        with torch.no_grad():
            out_after = fused(self.input)

        err = max_abs_error(out_before, out_after)
        self.assertLess(err, 1e-3, f"Fuse error too large: {err}")

    def test_param_reduction_after_fuse(self):
        model = copy.deepcopy(self.model)
        before = count_params(model)
        model.fuse_entire_model()
        after = count_params(model)
        self.assertLess(after, before, "Params should decrease after fuse")

    # ====================== Gradient ======================
    def test_gradient_flow(self):
        self.model.train()
        output, *_ = self.model(self.input, return_all=True)
        loss = output.mean()
        loss.backward()

        has_grad = any(p.grad is not None for p in self.model.parameters())
        self.assertTrue(has_grad)

    # ====================== Internal ======================
    def test_internal_decode(self):
        self.model.eval()
        with torch.no_grad():
            x = self.input
            s0 = self.model.shallow_extractor(x)
            skip1, e1 = self.model.enc1(s0)
            skip2, e2 = self.model.enc2(e1)
            skip3, e3 = self.model.enc3(e2)

            b_out = self.model.bottleneck(e3)
            output = self.model._decode(b_out, skip1, skip2, skip3, x)

        self.assertEqual(output.shape, (1, 3, 256, 256))

    # ====================== Early Exit ======================
    def test_early_exit(self):
        bottleneck = AdaptiveLoopedBottleneck(dim=128, max_loops=6).to(self.device)
        bottleneck.eval()

        x = torch.randn(1, 128, 32, 32, device=self.device)
        with torch.no_grad():
            out = bottleneck(x, halt_threshold=0.6)

        self.assertEqual(out.shape, x.shape)

    # ====================== Benchmark ======================
    def test_flops(self):
        self.model.eval()
        flops = FlopCountAnalysis(self.model, self.input).total()
        self.assertGreater(flops, 0)

    def test_latency(self):
        latency = benchmark_latency(self.model, self.input, self.device, warmup=5, iters=20)
        self.assertGreater(latency, 0)

# =========================================================
# Benchmark Summary
# =========================================================

def benchmark_summary():

    print("=" * 60)
    print("FULL MODEL BENCHMARK")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = LoopRepDocEnhanceNet(
        base_dim=32,
        max_loops=4
    ).to(device)

    x = torch.randn(
        1,
        3,
        256,
        256
    ).to(device)

    # -----------------------------------------

    params_before = count_params(model)

    model.eval()

    flops_before = (
        FlopCountAnalysis(model, x).total() / 1e9
    )

    print(f"Params before fuse : {params_before:,}")
    print(f"GFLOPs before fuse : {flops_before:.2f}")

    # -----------------------------------------

    model.fuse_entire_model()

    params_after = count_params(model)

    flops_after = (
        FlopCountAnalysis(model, x).total() / 1e9
    )

    print("-" * 60)

    print(f"Params after fuse  : {params_after:,}")
    print(f"GFLOPs after fuse  : {flops_after:.2f}")

    reduction = (
        (params_before - params_after)
        / params_before
    ) * 100

    print(f"Reduction          : {reduction:.2f}%")

    # -----------------------------------------

    cpu_latency = benchmark_latency(
        model, x, torch.device("cpu"), warmup=5, iters=20
    )
    cpu_fps = 1000 / cpu_latency

    print(f"CPU Latency        : {cpu_latency:.2f} ms")
    print(f"CPU FPS            : {cpu_fps:.2f} FPS")

    if torch.cuda.is_available():
        gpu_latency = benchmark_latency(
            model, x, torch.device("cuda"), warmup=20, iters=100
        )
        gpu_fps = 1000 / gpu_latency

        print(f"GPU Latency        : {gpu_latency:.2f} ms")
        print(f"GPU FPS            : {gpu_fps:.2f} FPS")

    print("=" * 60)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    benchmark_summary()

    print("\nRunning unit tests...\n")

    unittest.main(verbosity=2)