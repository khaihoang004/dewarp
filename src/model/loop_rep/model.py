import torch
import torch.nn as nn
import torch.nn.functional as F

from .repconv import RepConv3, RepConv7
from .attention import DocumentAttn, RestormerAttention


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return norm * self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

# --- 3. Tầng SwiGLU Feed-Forward Network (Tối Ưu Tham Số) ---
class SwiGLU_FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Giảm hệ số ẩn xuống 8/3 (~2.67) để bảo toàn lượng tham số của tầng FFN thông thường
        hidden_dim = int(dim * 8 / 3)
        
        self.w_v = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        # Chia đôi tensor đầu ra thành 2 nhánh W và V
        w, v = torch.chunk(self.w_v(x), 2, dim=1)
        # Công thức: Swish(w) * v
        return self.project_out(F.silu(w) * v)


class BottleneckLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = RestormerAttention(dim)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU_FFN(dim)
        
        # Gate chạy song song với mỗi layer để tính Exit Probability
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        
        # Tính xác suất exit tại bước này: dạng tensor (B, 1, 1, 1) -> view về (B,)
        exit_prob = self.gate(x).view(-1)
        return x, exit_prob


# --- KIẾN TRÚC LOOPED BOTTLENECK CHÍNH ---
class AdaptiveLoopedBottleneck(nn.Module):
    def __init__(self, dim, max_loops=4):
        super().__init__()
        self.max_loops = max_loops
        
        # Khởi tạo 1 Layer dùng chung trọng số (Weight-sharing dạng loop)
        self.layer = BottleneckLayer(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Nơi lưu trữ thông tin của các bước
        layer_outputs = []
        exit_probs = []
        
        state = x
        
        # 1. Chạy tuần tự qua các Layer (Reasoning Steps)
        for t in range(self.max_loops):
            state, p_t = self.layer(state)
            layer_outputs.append(state)
            exit_probs.append(p_t)
            
        # 2. Xử lý logic xác suất để tạo trọng số dừng (Halting Weights)
        # Sử dụng cơ chế như PonderNet / ACT
        halting_weights = []
        survival_prob = torch.ones(B, device=x.device) # S_0 = 1
        
        for t in range(self.max_loops):
            p_t = exit_probs[t]
            
            if t < self.max_loops - 1:
                # w_t = S_{t-1} * p_t
                w_t = survival_prob * p_t
                # Cập nhật S_t = S_{t-1} * (1 - p_t)
                survival_prob = survival_prob * (1.0 - p_t)
            else:
                # Bước cuối: Nhận toàn bộ lượng "mass" còn lại: w_T = S_{T-1}
                w_t = survival_prob
                
            halting_weights.append(w_t)
            
        # 3. Phân tách hành vi Train và Inference (Early Exit thực tế)
        if self.training:
            # LƯU Ý: Trong khi TRAIN, trả về list các output và trọng số dừng để tính Loss phân tầng
            # Output cuối cùng của Bottleneck là weighted sum của tất cả các bước
            final_out = torch.zeros_like(x)
            for t in range(self.max_loops):
                # Phát triển chiều của w_t để nhân với tensor 4D (B, C, H, W)
                w_expanded = halting_weights[t].view(B, 1, 1, 1)
                final_out = final_out + w_expanded * layer_outputs[t]
                
            return final_out, layer_outputs, halting_weights
        else:
            # LƯU Ý: Khi INFERENCE, thực hiện Early Exit cứng (Hard Exit) để tối ưu hóa thời gian chạy
            # Sử dụng vòng lặp thực tế để mô phỏng suy luận động
            state_infer = x
            survival_prob_infer = 1.0
            
            for t in range(self.max_loops):
                state_infer, p_t_infer = self.layer(state_infer)
                
                # Tính toán trọng số tích lũy tại runtime
                if t < self.max_loops - 1:
                    w_t_infer = survival_prob_infer * p_t_infer
                    survival_prob_infer = survival_prob_infer * (1.0 - p_t_infer)
                else:
                    w_t_infer = survival_prob_infer
                
                # Chiến lược Early Exit: Nếu xác suất dừng thực tế w_t đạt ngưỡng cao 
                # Hoặc p_t độc lập > 0.85 (tùy bạn cấu hình), ta dừng việc tính toán layer tiếp theo.
                if w_t_infer.mean() > 0.80:
                    break
                    
            return state_infer, None, None


class ShallowExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.conv = RepConv7(in_channels, out_channels, deploy=deploy)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))


class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.conv = RepConv3(in_channels, out_channels, deploy=deploy)
        self.act = nn.GELU()
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        feat = self.act(self.conv(x))
        downsampled = self.down(feat)
        return feat, downsampled


class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, deploy=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = RepConv3(in_channels + skip_channels, out_channels, deploy=deploy)
        self.act = nn.GELU()

    def forward(self, x, skip_feat):
        x = self.up(x)
        x = torch.cat([x, skip_feat], dim=1)
        return self.act(self.conv(x))


class LoopRepDocEnhanceNet(nn.Module):
    def __init__(self, base_dim=32, max_loops=4, deploy=False):
        super().__init__()
        self.base_dim = base_dim
        
        # Shallow Extraction (H, W) -> base_dim (Kênh 32)
        self.shallow_extractor = ShallowExtractor(3, base_dim, deploy=deploy)
        
        # Encoder Blocks (Hạ không gian từ 1/1 xuống 1/8)
        self.enc1 = EncoderStage(base_dim, base_dim, deploy=deploy)         # 1/1 -> 1/2
        self.enc2 = EncoderStage(base_dim, base_dim * 2, deploy=deploy)     # 1/2 -> 1/4
        self.enc3 = EncoderStage(base_dim * 2, base_dim * 4, deploy=deploy) # 1/4 -> 1/8
        
        # Bottleneck thích ứng lõi động (Chạy ở scale 1/8)
        self.bottleneck = AdaptiveLoopedBottleneck(dim=base_dim * 4, max_loops=max_loops)
        
        # Decoder Blocks (Khôi phục không gian từ 1/8 lên lại 1/1)
        self.dec3 = DecoderStage(base_dim * 4, base_dim * 4, base_dim * 2, deploy=deploy) # 1/8 -> 1/4
        self.dec2 = DecoderStage(base_dim * 2, base_dim * 2, base_dim, deploy=deploy)     # 1/4 -> 1/2
        self.dec1 = DecoderStage(base_dim, base_dim, base_dim, deploy=deploy)             # 1/2 -> 1/1
        
        # Output Head: Dự đoán phần dư toàn cục (Global Residual Mapping)
        self.final_head = nn.Conv2d(base_dim, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x_ori = x # Giữ lại ảnh gốc đầu vào
        
        # --- ENCODER PATH ---
        s0 = self.shallow_extractor(x)
        skip1, e1 = self.enc1(s0)
        skip2, e2 = self.enc2(e1)
        skip3, e3 = self.enc3(e2)
        
        # --- ADAPTIVE BOTTLENECK ---
        b_out, layer_outputs, halting_weights = self.bottleneck(e3)
            
        # --- DECODER PATH ---
        d3 = self.dec3(b_out, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)
        
        # --- RECONSTRUCTION ---
        residual = self.final_head(d1)
        x_final_output = x_ori + residual # Cộng phần dư đa nhiệm (Deblur, Deshadow, v.v.)
        
        if self.training:
            return x_final_output, layer_outputs, halting_weights
            
        return torch.clamp(x_final_output, 0.0, 1.0)

    @torch.no_grad()
    def fuse_entire_model(self):
        fused = []
        for name, module in self.named_modules():
            if (
                hasattr(module, 'deploy')
                and hasattr(module, 'fuse')
                and not module.deploy
                and name != ''
            ):
                module.fuse()
                fused.append(name)
 
        print(f">>> ĐÃ FUSE {len(fused)} MODULE: {fused}")



import unittest
import torch
import torch.nn as nn

# Giả sử file kiến trúc của bạn đặt tên là model.py
# from model import LoopRepDocEnhanceNet

class TestLoopRepDocEnhanceNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Khởi tạo cấu hình mạng và thiết bị chạy cố định cho toàn bộ bài test"""
        cls.base_dim = 32
        cls.max_loops = 4
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Khởi tạo mô hình
        cls.model = LoopRepDocEnhanceNet(
            base_dim=cls.base_dim, 
            max_loops=cls.max_loops, 
            deploy=False
        ).to(cls.device)
        
        # Giả lập 1 Batch ảnh tài liệu 2K (Batch_size=1, Kênh màu RGB=3, 512x512)
        # Sử dụng kích thước 2K thực tế để kiểm tra khả năng chịu tải và tính chia hết cho 8
        cls.dummy_input = torch.randn(1, 3, 512, 512, device=cls.device)

    def test_1_training_mode_output(self):
        """Kiểm thử luồng dữ liệu ở chế độ Huấn luyện (Training Mode)"""
        self.model.train()
        
        # Forward pass nhận bộ 3 đầu ra phân tầng
        final_output, layer_outputs, halting_weights = self.model(self.dummy_input)
        
        # 1. Kiểm tra ảnh đầu ra cuối cùng phải khớp hoàn hảo kích thước ảnh 2K gốc
        self.assertEqual(final_output.shape, (1, 3, 512, 512))
        
        # 2. Kiểm tra số lượng reasoning steps ẩn trong Bottleneck phải đúng bằng max_loops
        self.assertEqual(len(layer_outputs), self.max_loops)
        self.assertEqual(len(halting_weights), self.max_loops)
        
        # 3. Kiểm tra kích thước đặc trưng tại Bottleneck (độ phân giải giảm 1/8 -> 256x256, số kênh 4 * base_dim = 128)
        expected_bottleneck_shape = (1, self.base_dim * 4, 512 // 8, 512 // 8)
        for h_t in layer_outputs:
            self.assertEqual(h_t.shape, expected_bottleneck_shape)
            
        # 4. Kiểm tra chiều của mảng trọng số dừng (Halting weights) ứng với từng ảnh trong Batch
        for w_t in halting_weights:
            self.assertEqual(w_t.shape, (1,))

    def test_2_inference_mode_and_fusion(self):
        """Kiểm thử luồng dữ liệu ở chế độ Suy luận (Inference Mode) sau khi đã rút gọn mạng (Fuse)"""
        self.model.eval()
        
        # Tiến hành nén (fuse) các nhánh RepConv3 và RepConv7
        try:
            self.model.fuse_entire_model()
        except Exception as e:
            self.fail(f"Hàm fuse_entire_model() bị lỗi hệ thống: {e}")
            
        # Chạy suy luận không tính đạo hàm để tiết kiệm bộ nhớ
        with torch.no_grad():
            inference_output = self.model(self.dummy_input)
            
        # 1. Kiểm tra ảnh đầu ra sau khi fuse phải giữ nguyên cấu trúc không gian 2K
        self.assertEqual(inference_output.shape, (1, 3, 512, 512))
        
        # 2. Kiểm tra dải giá trị màu đầu ra đã được ép (clamp) chuẩn trong khoảng [0.0, 1.0] hay chưa
        self.assertTrue(torch.all(inference_output >= 0.0))
        self.assertTrue(torch.all(inference_output <= 1.0))
        
        # 3. Kiểm tra chế độ Eval chỉ trả về duy nhất 1 Tensor ảnh (không trả về tuple/list như khi train)
        self.assertIsInstance(inference_output, torch.Tensor)

if __name__ == "__main__":
    # Kích hoạt trình chạy test tự động của bộ framework
    unittest.main(verbosity=2)