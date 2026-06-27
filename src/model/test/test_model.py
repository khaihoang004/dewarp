import os
import time
import copy
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

# =========================================================
# CHÚ Ý: Import đúng đường dẫn tới các file bạn đã chia nhỏ
# =========================================================
from src.model.deshadow.model import DocDeshadowNet
from src.model.deshadow.blocks import AdaptiveLoopedBottleneck

ATOL = 1e-4

# =========================================================
# UTILS FUNCTIONS
# =========================================================

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

# =========================================================
# STANDALONE TEST FUNCTIONS 
# =========================================================

def test_training_forward(model, x, max_loops):
    model.train()
    B, C, H, W = x.shape
    output, inter_preds, halting_weights, gate_logits, exit_probs = model(x, return_all=True)
    
    assert output.shape == (B, C, H, W), f"Sai cấu trúc output: {output.shape}"
    assert len(inter_preds) == max_loops, f"Thiếu intermediate predictions: {len(inter_preds)}"
    assert len(halting_weights) == max_loops, f"Thiếu halting weights: {len(halting_weights)}"

def test_halting_weights_sum_to_one(model, x):
    model.train()
    _, _, halting_weights, _, _ = model(x, return_all=True)
    total = halting_weights.sum(dim=0)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5), "Tổng halting weights không bằng 1!"

def test_inference_forward(model, x):
    model.eval()
    B, C, H, W = x.shape
    with torch.no_grad():
        output = model(x)
    assert output.shape == (B, C, H, W), f"Sai cấu trúc output inference: {output.shape}"

def test_output_range(model, x):
    model.eval()
    with torch.no_grad():
        output = model(x)
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0), "Output nằm ngoài khoảng an toàn [0.0, 1.0]"

def test_fuse_consistency(model, x):
    model.eval()
    with torch.no_grad():
        out_before = model(x)

    fused = copy.deepcopy(model)
    fused.fuse_entire_model()

    with torch.no_grad():
        out_after = fused(x)

    err = max_abs_error(out_before, out_after)
    assert err < 1e-3, f"Sai số sau khi fuse quá lớn: {err}"

def test_param_reduction_after_fuse(model):
    model_cp = copy.deepcopy(model)
    before = count_params(model_cp)
    model_cp.fuse_entire_model()
    after = count_params(model_cp)
    assert after < before, f"Số lượng tham số không giảm sau khi Fuse! (Trước: {before}, Sau: {after})"

def test_gradient_flow(model, x):
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
            
    output, *_ = model(x, return_all=True)
    loss = output.mean()
    loss.backward()

    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "Đồ thị Gradient bị đứt gãy, không tìm thấy đạo hàm!"

def test_internal_decode(model, x):
    model.eval()
    B, C, H, W = x.shape
    with torch.no_grad():
        x_L2 = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_L3 = torch.nn.functional.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)

        feat_L1 = model.shallow_L1(x)
        skip1, down1 = model.enc1(feat_L1)
        
        feat_L2 = model.shallow_L2(x_L2)
        skip2, down2 = model.enc2(torch.cat([down1, feat_L2], dim=1))
        
        feat_L3 = model.shallow_L3(x_L3)
        skip3, down3 = model.enc3(torch.cat([down2, feat_L3], dim=1))

        b_out = model.bottleneck(down3)
        
        # Mô phỏng quá trình decode như trong hàm forward của Ultimate model
        d3 = model.dec3(b_out, skip3)
        d2 = model.dec2(d3, skip2)
        d1 = model.dec1(d2, skip1)
        residual = model.final_head(d1)
        output = x + residual
        
    assert output.shape == (B, C, H, W), f"Luồng decode nội bộ sai shape: {output.shape}"

def test_early_exit(device):
    # Khởi tạo kích thước theo dim=128 như test mẫu của bạn
    bottleneck = AdaptiveLoopedBottleneck(dim=128, max_loops=6).to(device)
    bottleneck.eval()

    x = torch.randn(1, 128, 32, 32, device=device)
    with torch.no_grad():
        out = bottleneck(x, halt_threshold=0.6)

    assert out.shape == x.shape, f"Early exit đổi kích thước đặc trưng: {out.shape}"

def test_flops(model, x):
    model.eval()
    flops = FlopCountAnalysis(model, x).total()
    assert flops > 0, "Không tính toán được FLOPs"


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
    
    model = DocDeshadowNet(
        base_dim=base_dim,
        max_loops=max_loops,
        deploy=False
    ).to(device)
    
    tests = [
        ("Training Forward Pass", lambda: test_training_forward(model, input_tensor, max_loops)),
        ("Halting Weights Consistency", lambda: test_halting_weights_sum_to_one(model, input_tensor)),
        ("Inference Forward Pass", lambda: test_inference_forward(model, input_tensor)),
        ("Output Range Clamping", lambda: test_output_range(model, input_tensor)),
        ("Fuse Mathematical Consistency", lambda: test_fuse_consistency(model, input_tensor)),
        ("Parameter Reduction via RepVGG Fuse", lambda: test_param_reduction_after_fuse(model)),
        ("Backward Gradient Flow", lambda: test_gradient_flow(model, input_tensor)),
        ("Internal U-Net Decoder Mapping", lambda: test_internal_decode(model, input_tensor)),
        ("Adaptive Looped Early Exit", lambda: test_early_exit(device)),
        ("Flops Analysis Access", lambda: test_flops(model, input_tensor)),
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
    model = DocDeshadowNet(base_dim=32, max_loops=4).to(device)
    
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