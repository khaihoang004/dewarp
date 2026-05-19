"""
full_model_test.py

Bộ test tổng hợp cho:

1. LoopRepDocEnhanceNet
2. AdaptiveLoopedBottleneck
3. Fuse consistency
4. Early Exit
5. Gradient flow
6. ACT halting weights
7. Internal _decode func
8. FLOPs / Params / Latency benchmark

Chạy:
    pytest full_model_test.py -v

Hoặc:
    python full_model_test.py
"""

import copy
import time
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore.nn import FlopCountAnalysis

from src.model.loop_rep.model import (
    LoopRepDocEnhanceNet,
    AdaptiveLoopedBottleneck
)


# =========================================================
# Utils
# =========================================================

ATOL = 1e-4


def max_abs_error(a, b):
    return (a - b).abs().max().item()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def benchmark_latency(
    model,
    dummy_input,
    device,
    warmup=10,
    iters=50
):
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


# =========================================================
# Main Test
# =========================================================

class TestLoopRepDocEnhanceNet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        torch.manual_seed(42)

        cls.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        cls.base_dim = 32
        cls.max_loops = 4

        cls.model = LoopRepDocEnhanceNet(
            base_dim=cls.base_dim,
            max_loops=cls.max_loops,
            deploy=False
        ).to(cls.device)

        cls.input = torch.randn(
            1,
            3,
            256,
            256,
            device=cls.device
        )

    # =====================================================
    # training forward
    # =====================================================

    def test_training_forward(self):

        self.model.train()

        (
            output,
            intermediate_preds,
            halting_weights,
            halt_logits
        ) = self.model(self.input, return_all=True)

        self.assertEqual(
            output.shape,
            (1, 3, 256, 256)
        )

        self.assertEqual(
            len(intermediate_preds),
            self.max_loops
        )

        self.assertEqual(
            len(halting_weights),
            self.max_loops
        )

        self.assertEqual(
            len(halt_logits),
            self.max_loops
        )

    # =====================================================
    # intermediate preds shape
    # =====================================================

    def test_intermediate_preds_shapes(self):

        self.model.train()

        (
            output,
            intermediate_preds,
            halting_weights,
            halt_logits
        ) = self.model(self.input, return_all=True)

        expected_shape = (
            1,
            3,
            256,
            256
        )

        for p in intermediate_preds:
            self.assertEqual(p.shape, expected_shape)

        for w in halting_weights:
            self.assertEqual(
                w.shape,
                (1,)  # Batch size = 1
            )

    # =====================================================
    # halting weights sum
    # =====================================================

    def test_halting_weights_sum_to_one(self):

        self.model.train()

        (
            output,
            _,
            halting_weights,
            halt_logits
        ) = self.model(self.input, return_all=False)

        weights = halting_weights

        summed = weights.sum(dim=0)

        err = max_abs_error(
            summed,
            torch.ones_like(summed)
        )

        self.assertLess(err, 1e-5)

    # =====================================================
    # inference forward
    # =====================================================

    def test_inference_forward(self):

        self.model.eval()

        with torch.no_grad():
            output = self.model(self.input)

        self.assertIsInstance(
            output,
            torch.Tensor
        )

        self.assertEqual(
            output.shape,
            (1, 3, 256, 256)
        )

    # =====================================================
    # clamp range
    # =====================================================

    def test_output_range(self):

        self.model.eval()

        with torch.no_grad():
            output = self.model(self.input)

        self.assertTrue(
            torch.all(output >= 0.0)
        )

        self.assertTrue(
            torch.all(output <= 1.0)
        )

    # =====================================================
    # fuse consistency
    # =====================================================

    def test_fuse_consistency(self):

        self.model.eval()

        with torch.no_grad():
            out_before = self.model(self.input)

        fused_model = copy.deepcopy(self.model)

        fused_model.fuse_entire_model()

        fused_model.eval()

        with torch.no_grad():
            out_after = fused_model(self.input)

        err = max_abs_error(
            out_before,
            out_after
        )

        self.assertLess(
            err,
            1e-3
        )

    # =====================================================
    # param reduction after fuse
    # =====================================================

    def test_param_reduction_after_fuse(self):

        model = copy.deepcopy(self.model)

        before = count_params(model)

        model.fuse_entire_model()

        after = count_params(model)

        self.assertLess(after, before)

    # =====================================================
    # gradient flow
    # =====================================================

    def test_gradient_flow(self):

        self.model.train()

        (
            output,
            intermediate_preds,
            halting_weights,
            halt_logits
        ) = self.model(self.input, return_all=True)

        loss = output.mean()

        loss.backward()

        has_grad = False

        for p in self.model.parameters():

            if p.grad is not None:
                has_grad = True
                break

        self.assertTrue(has_grad)

    # =====================================================
    # internal decode function
    # =====================================================

    def test_internal_decode(self):

        self.model.eval()

        with torch.no_grad():

            x = self.input

            s0 = self.model.shallow_extractor(x)

            skip1, e1 = self.model.enc1(s0)

            skip2, e2 = self.model.enc2(e1)

            skip3, e3 = self.model.enc3(e2)

            bottleneck_out = self.model.bottleneck(e3)

            output = self.model._decode(
                bottleneck_out,
                skip1,
                skip2,
                skip3,
                x
            )

        self.assertEqual(
            output.shape,
            (1, 3, 256, 256)
        )

    # =====================================================
    # early exit
    # =====================================================

    def test_early_exit(self):

        bottleneck = AdaptiveLoopedBottleneck(
            dim=128,
            max_loops=6
        ).to(self.device)

        bottleneck.eval()

        x = torch.randn(
            1,
            128,
            32,
            32,
            device=self.device
        )

        with torch.no_grad():

            out = bottleneck(
                x,
                halt_threshold=0.5
            )

        self.assertEqual(
            out.shape,
            x.shape
        )

    # =====================================================
    # flops
    # =====================================================

    def test_flops(self):

        self.model.eval()

        flops = FlopCountAnalysis(
            self.model,
            self.input
        )

        total = flops.total()

        self.assertGreater(total, 0)

    # =====================================================
    # benchmark
    # =====================================================

    def test_latency(self):

        self.model.eval()

        latency = benchmark_latency(
            self.model,
            self.input,
            self.device,
            warmup=3,
            iters=10
        )

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
        model,
        x,
        torch.device("cpu"),
        warmup=3,
        iters=10
    )

    print("-" * 60)

    print(f"CPU latency        : {cpu_latency:.2f} ms")

    if torch.cuda.is_available():

        gpu_latency = benchmark_latency(
            model,
            x,
            torch.device("cuda"),
            warmup=10,
            iters=50
        )

        fps = 1000 / gpu_latency

        print(f"GPU latency        : {gpu_latency:.2f} ms")
        print(f"FPS                : {fps:.2f}")

    print("=" * 60)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    benchmark_summary()

    print("\nRunning unit tests...\n")

    unittest.main(verbosity=2)