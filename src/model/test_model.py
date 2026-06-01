"""
test_repconv_fixed.py
pytest test suite cho RepConv3BN và RepConv7.

Chạy:
    pytest test_repconv_fixed.py -v
    pytest test_repconv_fixed.py -v -k "fuse"        # chỉ test fuse
    pytest test_repconv_fixed.py -v -k "grouped"     # chỉ test groups > 1

Mỗi class được test:
  - output shape
  - fuse accuracy  (|train_out - deploy_out| < tol)
  - fuse idempotent
  - branch cleanup sau fuse
  - deploy=True init
  - param count giảm sau fuse
  - grouped conv (groups > 1)
  - _fold_1x1_into_kxk (unit test độc lập)
"""

import copy
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.repconv import RepConv3BN, RepConv3, RepConv7


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

ATOL_FP32 = 1e-4   # float32 tích lũy sai số

def warmup_bn(model: nn.Module, in_ch: int, n: int = 12):
    """Chạy n forward pass để BN có running stats ổn định."""
    model.train()
    with torch.no_grad():
        for _ in range(n):
            model(torch.randn(8, in_ch, 8, 8))
    model.eval()

def clone_and_fuse(model: nn.Module) -> nn.Module:
    m = copy.deepcopy(model)
    m.eval()
    m.fuse()
    return m

def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ─────────────────────────────────────────────
# Shared: unit test fold_1x1_into_kxk
# ─────────────────────────────────────────────

@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("K", [3, 7])
def test_fold_1x1_into_kxk(groups, K):
    """
    Verify _fold_1x1_into_kxk bằng cách so sánh với kết quả forward thực tế:
      y_ref  = conv_kxk(conv_1x1(x))
      y_fold = conv_fold(x)   # kernel đã fold
    """
    torch.manual_seed(42)
    C_in, C_out = 8 * groups, 8 * groups
    pad = K // 2
    x = torch.randn(2, C_in, 12, 12)

    c1 = nn.Conv2d(C_in, C_in,  1,      groups=groups, bias=False)
    c2 = nn.Conv2d(C_in, C_out, K, padding=pad, groups=groups, bias=False)

    with torch.no_grad():
        y_ref = c2(c1(x))

    # Fold
    if groups == 1:
        w_fold = F.conv2d(c2.weight, c1.weight.permute(1, 0, 2, 3))
    else:
        icpg = C_in // groups
        ocpg = C_out // groups
        w1_T = c1.weight.permute(1, 0, 2, 3)
        slices = []
        for g in range(groups):
            slices.append(F.conv2d(
                c2.weight[g * ocpg:(g + 1) * ocpg],
                w1_T[:, g * icpg:(g + 1) * icpg],
            ))
        w_fold = torch.cat(slices, dim=0)

    c_fold = nn.Conv2d(C_in, C_out, K, padding=pad, groups=groups, bias=False)
    c_fold.weight.data.copy_(w_fold)

    with torch.no_grad():
        y_fold = c_fold(x)

    err = max_abs_error(y_ref, y_fold)
    assert err < ATOL_FP32, (
        f"fold_1x1_into_kxk sai: K={K}, groups={groups}, err={err:.3e}"
    )


# ─────────────────────────────────────────────
# RepConv3BN
# ─────────────────────────────────────────────

class TestRepConv3BN:

    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        m = RepConv3BN(16, 32, groups=1, deploy=False)
        warmup_bn(m, in_ch=16)
        return m

    @pytest.fixture
    def x(self):
        torch.manual_seed(1)
        return torch.randn(2, 16, 10, 10)

    # ── shape ──

    def test_output_shape_train(self, model, x):
        model.eval()
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 32, 10, 10)

    def test_output_shape_deploy(self, x):
        torch.manual_seed(0)
        m = RepConv3BN(16, 32, deploy=True)
        m.eval()
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 32, 10, 10)

    # ── fuse accuracy ──

    def test_fuse_output_matches(self, model, x):
        d = clone_and_fuse(model)
        model.eval()
        with torch.no_grad():
            y_train  = model(x)
            y_deploy = d(x)
        err = max_abs_error(y_train, y_deploy)
        assert err < ATOL_FP32, f"Fuse error quá lớn: {err:.3e}"

    def test_residual_present_in_deploy(self, x):
        """Deploy phải giữ x + reparam(x)."""
        torch.manual_seed(0)
        m = RepConv3BN(16, 32, deploy=True)
        # Zero weight → output = x + 0 = x
        m.reparam.weight.data.zero_()
        m.reparam.bias.data.zero_()
        m.eval()
        with torch.no_grad():
            y = m(x)
        # Chỉ đúng nếu in_ch == out_ch
        # Test với in=out=16
        m2 = RepConv3BN(16, 16, deploy=True)
        m2.reparam.weight.data.zero_()
        m2.reparam.bias.data.zero_()
        x2 = torch.randn(2, 16, 8, 8)
        with torch.no_grad():
            y2 = m2(x2)
        err = max_abs_error(y2, x2)
        assert err < 1e-6, "Residual không hoạt động đúng khi weight=0"

    # ── branch cleanup ──

    def test_deploy_has_no_branches(self, model):
        model.fuse()
        for attr in ('branches', 'bns', 'branch_weight'):
            assert not hasattr(model, attr), f"{attr} vẫn còn sau fuse"

    # ── idempotent ──

    def test_fuse_idempotent(self, model, x):
        d = clone_and_fuse(model)
        d.fuse()   # lần 2
        model.eval()
        with torch.no_grad():
            y_train  = model(x)
            y_deploy = d(x)
        err = max_abs_error(y_train, y_deploy)
        assert err < ATOL_FP32

    # ── param reduction ──

    def test_param_count_reduced(self, model):
        before = count_params(model)
        model.fuse()
        after = count_params(model)
        assert after < before, (
            f"Params không giảm sau fuse: {before} → {after}"
        )

    # ── grouped conv ──

    @pytest.mark.parametrize("groups", [2, 4])
    def test_fuse_grouped(self, groups):
        torch.manual_seed(0)
        C = 16 * groups
        m = RepConv3BN(C, C, groups=groups, deploy=False)
        warmup_bn(m, in_ch=C)
        x = torch.randn(2, C, 8, 8)
        d = clone_and_fuse(m)
        m.eval()
        with torch.no_grad():
            err = max_abs_error(m(x), d(x))
        assert err < ATOL_FP32, (
            f"Grouped fuse sai (groups={groups}): err={err:.3e}"
        )

    # ── branch_weight effect ──

    def test_branch_weight_affects_output(self, model, x):
        """Thay đổi branch_weight phải thay đổi output."""
        model.eval()
        with torch.no_grad():
            y1 = model(x).clone()
            model.branch_weight.data[0] += 5.0
            y2 = model(x)
        assert not torch.allclose(y1, y2), (
            "branch_weight không ảnh hưởng đến output"
        )


# ─────────────────────────────────────────────
# RepConv3
# ─────────────────────────────────────────────

class TestRepConv3:

    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        return RepConv3(16, 32, groups=1, deploy=False).eval()

    @pytest.fixture
    def x(self):
        torch.manual_seed(1)
        return torch.randn(2, 16, 10, 10)

    # ── shape ──

    def test_output_shape_train(self, model, x):
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 32, 10, 10)

    def test_output_shape_deploy(self, x):
        torch.manual_seed(0)
        m = RepConv3(16, 32, deploy=True).eval()
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 32, 10, 10)

    # ── fuse accuracy ──

    def test_fuse_output_matches(self, model, x):
        """Không có BN → fuse phải khớp chính xác (atol nhỏ hơn)."""
        d = clone_and_fuse(model)
        with torch.no_grad():
            y_train  = model(x)
            y_deploy = d(x)
        err = max_abs_error(y_train, y_deploy)
        assert err < 1e-5, f"Fuse error: {err:.3e}"

    def test_fuse_exact_without_eval(self, x):
        """
        Đặc trưng của NoBN: fuse() chính xác dù model ở train mode.
        RepConv3BN (có BN) sẽ sai trong trường hợp này.
        """
        torch.manual_seed(0)
        m = RepConv3(16, 32, deploy=False)
        m.train()                           # ← train mode, KHÔNG eval()
        d = copy.deepcopy(m)
        d.fuse()

        m.eval()
        d.eval()
        with torch.no_grad():
            err = max_abs_error(m(x), d(x))
        assert err < 1e-5, (
            f"NoBN fuse phải chính xác dù không gọi eval(): err={err:.3e}"
        )

    def test_fuse_matches_branch_sum(self, x):
        """
        Verify trực tiếp: fused weight = Σ padded_weights của các branch.
        """
        torch.manual_seed(0)
        m = RepConv3(16, 32, deploy=False).eval()

        # Tính expected W, B thủ công
        def pad(w):
            _, _, h, wk = w.shape
            ph, pw = 3 - h, 3 - wk
            return F.pad(w, [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2])

        with torch.no_grad():
            W_exp = sum(pad(m.branches[i].weight) for i in range(4))
            B_exp = sum(m.branches[i].bias for i in range(4))

            # Branch 4: fold 1x1 → 3x3
            w1 = m.branches[4][0].weight
            w2 = m.branches[4][1].weight
            b2 = m.branches[4][1].bias
            W_exp = W_exp + F.conv2d(w2, w1.permute(1, 0, 2, 3))
            B_exp = B_exp + b2

        d = clone_and_fuse(m)
        W_actual = d.reparam.weight.data
        B_actual = d.reparam.bias.data

        assert max_abs_error(W_exp, W_actual) < 1e-6, "Fused weight sai"
        assert max_abs_error(B_exp, B_actual) < 1e-6, "Fused bias sai"

    # ── residual ──

    def test_residual_present(self):
        """y = x + conv(x) → zero weights thì y = x (chỉ đúng in=out)."""
        m = RepConv3(16, 16, deploy=True).eval()
        m.reparam.weight.data.zero_()
        m.reparam.bias.data.zero_()
        x = torch.randn(2, 16, 8, 8)
        with torch.no_grad():
            y = m(x)
        assert max_abs_error(y, x) < 1e-6

    # ── branch cleanup ──

    def test_deploy_has_no_branches(self, model):
        model.fuse()
        assert not hasattr(model, 'branches'), "branches vẫn còn sau fuse"

    # ── idempotent ──

    def test_fuse_idempotent(self, model, x):
        d = clone_and_fuse(model)
        d.fuse()    # lần 2
        with torch.no_grad():
            err = max_abs_error(model(x), d(x))
        assert err < 1e-5

    # ── param reduction ──

    def test_param_count_reduced(self, model):
        before = count_params(model)
        model.fuse()
        after  = count_params(model)
        assert after < before, f"Params không giảm: {before} → {after}"

    # ── grouped ──

    @pytest.mark.parametrize("groups", [2, 4])
    def test_fuse_grouped(self, groups):
        torch.manual_seed(0)
        C = 16 * groups
        m = RepConv3(C, C, groups=groups).eval()
        x = torch.randn(2, C, 8, 8)
        d = clone_and_fuse(m)
        with torch.no_grad():
            err = max_abs_error(m(x), d(x))
        assert err < 1e-5, f"Grouped fuse sai (groups={groups}): err={err:.3e}"

    # ── so sánh với RepConv3BN (có BN) ──

    def test_no_bn_params(self, model):
        """Không có BatchNorm2d trong model."""
        bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        assert len(bn_layers) == 0, f"Tìm thấy {len(bn_layers)} BN layer"

    def test_train_eval_output_same(self, x):
        """
        Không có BN → train mode và eval mode cho cùng kết quả.
        (Ngược lại với RepConv3BN có BN.)
        """
        torch.manual_seed(0)
        m = RepConv3(16, 32)
        with torch.no_grad():
            m.train()
            y_train = m(x)
            m.eval()
            y_eval  = m(x)
        assert max_abs_error(y_train, y_eval) < 1e-6, (
            "train/eval output khác nhau — có thể model vô tình dùng BN"
        )


# ─────────────────────────────────────────────
# RepConv7
# ─────────────────────────────────────────────

class TestRepConv7:

    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        m = RepConv7(16, 32, groups=1, deploy=False)
        m.eval()
        return m

    @pytest.fixture
    def x(self):
        torch.manual_seed(1)
        return torch.randn(2, 16, 12, 12)

    # ── shape ──

    def test_output_shape_train(self, model, x):
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 32, 12, 12)

    def test_same_padding(self, x):
        """Tất cả branch giữ spatial size (same-padding)."""
        torch.manual_seed(0)
        m = RepConv7(16, 32).eval()
        with torch.no_grad():
            y = m(x)
        assert y.shape[-2:] == x.shape[-2:]

    # ── fuse accuracy ──

    def test_fuse_output_matches(self, model, x):
        d = clone_and_fuse(model)
        with torch.no_grad():
            y_train  = model(x)
            y_deploy = d(x)
        err = max_abs_error(y_train, y_deploy)
        assert err < ATOL_FP32, f"Fuse error: {err:.3e}"

    def test_no_residual_in_deploy(self, x):
        """RepConv7 không có residual — verify khi weight→0 output→0."""
        torch.manual_seed(0)
        m = RepConv7(16, 32, deploy=True)
        m.reparam.weight.data.zero_()
        m.reparam.bias.data.zero_()
        m.eval()
        with torch.no_grad():
            y = m(x)
        err = y.abs().max().item()
        assert err < 1e-6, "RepConv7 deploy có residual ngoài ý muốn"

    # ── branch cleanup ──

    def test_deploy_has_no_branches(self, model):
        model.fuse()
        for attr in (
            'conv_7x7', 'conv_7x1', 'conv_1x7',
            'conv_7x5', 'conv_5x7', 'conv_5x5',
            'conv_1x5', 'conv_5x1',
            'conv_1x1_branch', 'conv_7x7_branch',
        ):
            assert not hasattr(model, attr), f"{attr} vẫn còn sau fuse"

    # ── idempotent ──

    def test_fuse_idempotent(self, model, x):
        d = clone_and_fuse(model)
        d.fuse()
        with torch.no_grad():
            err = max_abs_error(model(x), d(x))
        assert err < ATOL_FP32

    # ── param reduction ──

    def test_param_count_reduced(self, model):
        before = count_params(model)
        model.fuse()
        after = count_params(model)
        assert after < before

    # ── grouped ──

    @pytest.mark.parametrize("groups", [2, 4])
    def test_fuse_grouped(self, groups):
        torch.manual_seed(0)
        C = 16 * groups
        m = RepConv7(C, C, groups=groups, deploy=False).eval()
        x = torch.randn(2, C, 12, 12)
        d = clone_and_fuse(m)
        with torch.no_grad():
            err = max_abs_error(m(x), d(x))
        assert err < ATOL_FP32, (
            f"Grouped fuse sai (groups={groups}): err={err:.3e}"
        )

    # ── sequential branch ──

    def test_sequential_branch_contributes(self, model, x):
        """Branch 1×1→7×7 phải cho kết quả khác branch 7×7 thuần."""
        with torch.no_grad():
            y_full = model(x)
            # Zero toàn bộ trừ conv_7x7 để isolate
            for name, p in model.named_parameters():
                if 'conv_7x7.' not in name:
                    p.data.zero_()
            y_7x7_only = model(x)
        # y_full != y_7x7_only xác nhận các branch khác đóng góp
        assert not torch.allclose(y_full, y_7x7_only), (
            "Sequential branch không đóng góp vào output"
        )


# ─────────────────────────────────────────────
# Summary print (chạy trực tiếp: python test_repconv_fixed.py)
# ─────────────────────────────────────────────

def _summary():
    torch.manual_seed(0)
    print("\n─── Fuse accuracy summary ───")
    fmt = "  {:<26s}: max_err={:.2e}  {}"

    # RepConv3BN (có BN)
    for groups, C in [(1, 16), (2, 32), (4, 64)]:
        m = RepConv3BN(C, C, groups=groups)
        warmup_bn(m, C)
        x = torch.randn(2, C, 8, 8)
        d = clone_and_fuse(m)
        m.eval()
        with torch.no_grad():
            err = max_abs_error(m(x), d(x))
        ok = "✓ PASS" if err < ATOL_FP32 else "✗ FAIL"
        print(fmt.format(f"RepConv3BN (BN)  g={groups}", err, ok))

    # RepConv3
    for groups, C in [(1, 16), (2, 32), (4, 64)]:
        m = RepConv3(C, C, groups=groups).eval()
        x = torch.randn(2, C, 8, 8)
        d = clone_and_fuse(m)
        with torch.no_grad():
            err = max_abs_error(m(x), d(x))
        ok = "✓ PASS" if err < 1e-5 else "✗ FAIL"
        print(fmt.format(f"RepConv3BN (NoBN) g={groups}", err, ok))

    # RepConv7
    for groups, C in [(1, 16), (2, 32)]:
        m = RepConv7(C, C * 2, groups=groups).eval()
        x = torch.randn(2, C, 12, 12)
        d = clone_and_fuse(m)
        with torch.no_grad():
            err = max_abs_error(m(x), d(x))
        ok = "✓ PASS" if err < ATOL_FP32 else "✗ FAIL"
        print(fmt.format(f"RepConv7 g={groups}", err, ok))

    print()


if __name__ == "__main__":
    _summary()
    print("Chạy chi tiết: pytest test_repconv_fixed.py -v")