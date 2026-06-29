import os
import time
import copy
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

from src.deshadow.model.model import LoopRepDocEnhanceNet, AdaptiveLoopedBottleneck

ATOL = 1e-4

# =========================================================
# UTILS FUNCTIONS
# =========================================================

def max_abs_error(a, b):
    return (a - b).abs().max().item()

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def benchmark_latency(model, dummy_input, device, warmup=5, iters=20):
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
# STANDALONE TEST FUNCTIONS 
# =========================================================

def test_training_forward(model, x, max_loops):
    model.train()
    model.apply(lambda m: m.eval() if isinstance(m, nn.BatchNorm2d) else None)

    with torch.no_grad():
        res = model(x, calc_loss=True)

    output = res["final"]
    inter_preds = res["intermediate"]
    exit_logits = res["exit_logits"]
    exit_prob = res["exit_prob"]

    B, C, H, W = x.shape
    assert output.shape == (B, C, H, W)
    assert len(inter_preds) == max_loops
    assert exit_logits.shape[0] == max_loops

def test_halting_weights_sum_to_one(model, x):
    model.train()

    with torch.no_grad():
        res = model(x, calc_loss=True)

    h = res["halting"]

    assert torch.allclose(
        h.sum(dim=0),
        torch.ones_like(h.sum(dim=0)),
        atol=1e-5,
    )


def test_exit_prob_range(model, x):
    model.train()

    with torch.no_grad():
        res = model(x, calc_loss=True)

    p = res["exit_prob"]

    assert torch.all(p >= 0)
    assert torch.all(p <= 1)

    assert p.shape[1] == x.shape[0]

    
def test_inference_forward(model, x):
    model.eval()
    with torch.no_grad():
        res = model(x) # Trả về dict
        # Kiểm tra xem trả về là dict hay tensor
        output = res["final"] if isinstance(res, dict) else res

    B, C, H, W = x.shape
    assert output.shape == (B, C, H, W)

def test_output_range(model, x):
    model.eval()
    with torch.no_grad():
        res = model(x)
        output = res["final"] if isinstance(res, dict) else res
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0)

def test_fuse_consistency(model, x):
    model.eval()

    def get_output(m, input_x):
        res = m(input_x)
        return res["final"] if isinstance(res, dict) else res

    with torch.no_grad():
        out_before = get_output(model, x)

    fused = copy.deepcopy(model)
    fused.fuse_entire_model()

    with torch.no_grad():
        out_after = get_output(fused, x)

    err = max_abs_error(out_before, out_after)
    assert err < 1e-3, f"Fuse error too large: {err}"

def test_param_reduction_after_fuse(model):
    model_cp = copy.deepcopy(model)
    before = count_params(model_cp)
    model_cp.fuse_entire_model()
    after = count_params(model_cp)
    assert after < before, f"Số lượng tham số không giảm sau khi Fuse! (Trước: {before}, Sau: {after})"

def test_gradient_flow(model, x):
    model.train()
    model.apply(lambda m: m.eval() if isinstance(m, nn.BatchNorm2d) else None)

    res = model(x, calc_loss=True)
    loss = res["final"].mean() 

    loss.backward()

    has_grad = any(p.grad is not None and p.requires_grad for p in model.parameters())
    assert has_grad, "Gradient bị đứt"

def test_internal_decode(model, x):
    model.eval()
    B, C, H, W = x.shape
    device = x.device
    model = model.to(device)

    with torch.no_grad():
        s0 = model.shallow_extractor(x)

        skip1, e1 = model.enc1(s0)
        skip2, e2 = model.enc2(e1)
        skip3, e3 = model.enc3(e2)

        bottleneck_out = model.bottleneck(e3, calc_loss=True)

        b_out = bottleneck_out["final_state"]

        d3 = model.dec3(b_out, skip3)
        d2 = model.dec2(d3, skip2)
        d1 = model.dec1(d2, skip1)

        out = model.final_head(d1)

    assert out.shape == (B, C, H, W)


def test_early_exit(device):
    bottleneck = AdaptiveLoopedBottleneck(
        dim=128,
        max_loops=6,
    ).to(device)

    bottleneck.eval()

    x = torch.randn(1, 128, 32, 32, device=device)

    with torch.no_grad():
        out = bottleneck(
            x,
            halt_threshold=0.6,
        )

    final_state = out["final_state"]
    exit_steps = out["exit_steps"]

    print(f"Exit steps: {exit_steps.tolist()}")

    assert final_state.shape == x.shape
    assert exit_steps.shape == (1,)
    assert 1 <= exit_steps.item() <= bottleneck.max_loops


def test_flops(model, x):
    model.eval()

    with torch.no_grad():
        flops = FlopCountAnalysis(model, x).total()

    assert flops > 0


# =========================================================
# ORCHESTRATOR FUNCTION
# =========================================================

def run_all_tests(base_dim=32, max_loops=4, input_size=512):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
        
    print("=" * 60)
    print(f"RUNNING FUNCTIONAL UNIT TESTS | Target Size: {input_size[0]}x{input_size[1]}")
    print("=" * 60)
    
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_tensor = torch.randn(1, 3, input_size[0], input_size[1], device=device)
    
    model = LoopRepDocEnhanceNet(
        base_dim=base_dim,
        max_loops=max_loops,
        deploy=False
    ).to(device)
    
    tests = [
        ("Training Forward Pass", lambda: test_training_forward(model, input_tensor, max_loops)),
        ("Exit Prob Range", lambda: test_exit_prob_range(model, input_tensor)),
        ("Inference Forward Pass", lambda: test_inference_forward(model, input_tensor)),
        ("Output Range Clamping", lambda: test_output_range(model, input_tensor)),
        ("Fuse Consistency", lambda: test_fuse_consistency(model, input_tensor)),
        ("Parameter Reduction", lambda: test_param_reduction_after_fuse(model)),
        ("Gradient Flow", lambda: test_gradient_flow(model, input_tensor)),
        ("Internal Decode", lambda: test_internal_decode(model, input_tensor)),
        ("Early Exit", lambda: test_early_exit(device)),
        ("FLOPs", lambda: test_flops(model, input_tensor)),
    ]
        
    passed_counts = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"[PASSED] {name}")
            passed_counts += 1
        except AssertionError as e:
            print(f"[FAILED] {name} -> {e}")
        except Exception as e:
            print(f"[ERROR ] {name} đột ngột crash -> {e}")
            
    print("-" * 60)
    print(f"Result: {passed_counts}/{len(tests)} tests passed.")
    print("=" * 60 + "\n")


# =========================================================
# BENCHMARK SUMMARY 
# =========================================================

def benchmark_summary(input_size=512):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    print("=" * 60)
    print(f"FULL MODEL BENCHMARK | Target Size: {input_size[0]}x{input_size[1]}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LoopRepDocEnhanceNet(base_dim=32, max_loops=4).to(device)
    
    x = torch.randn(1, 3, input_size[0], input_size[1]).to(device)

    params_before = count_params(model)
    model.eval()
    flops_before = FlopCountAnalysis(model, x).total() / 1e9

    print(f"Params before fuse : {params_before:,}")
    print(f"GFLOPs before fuse : {flops_before:.2f}")

    model.fuse_entire_model()

    params_after = count_params(model)
    flops_after = FlopCountAnalysis(model, x).total() / 1e9

    print("-" * 60)
    print(f"Params after fuse  : {params_after:,}")
    print(f"GFLOPs after fuse  : {flops_after:.2f}")

    reduction = ((params_before - params_after) / params_before) * 100
    print(f"Reduction          : {reduction:.2f}%")
    print("-" * 60)

    cpu_latency = benchmark_latency(model, x, torch.device("cpu"), warmup=5, iters=20)
    cpu_fps = 1000 / cpu_latency
    print(f"CPU Latency        : {cpu_latency:.2f} ms")
    print(f"CPU FPS            : {cpu_fps:.2f} FPS")

    if torch.cuda.is_available():
        gpu_latency = benchmark_latency(model, x, torch.device("cuda"), warmup=20, iters=100)
        gpu_fps = 1000 / gpu_latency
        print(f"GPU Latency        : {gpu_latency:.2f} ms")
        print(f"GPU FPS            : {gpu_fps:.2f} FPS")

    print("=" * 60 + "\n")


# =========================================================
# MAIN EXECUTION ENTRY
# =========================================================

if __name__ == "__main__":
    TARGET_SIZE = (512, 512) 
    
    benchmark_summary(input_size=TARGET_SIZE)
    run_all_tests(base_dim=32, max_loops=4, input_size=TARGET_SIZE)