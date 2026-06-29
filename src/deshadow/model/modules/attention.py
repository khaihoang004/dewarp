import torch
import torch.nn as nn
import torch.nn.functional as F

class DocumentAttn(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Channel Attention
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention (Dùng Conv 3x3 thay vì 7x7 để tiết kiệm CPU)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.channel(x)
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weight = self.spatial(torch.cat([avg_out, max_out], dim=1))
        
        return x * spatial_weight

class RestormerAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_dwconv(self.qkv_conv(x))
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).view(B, C, H, W)
        return self.project_out(out)
    
try:
    from src.monarch_attn import MonarchAttention
except ImportError:
    MonarchAttention = None



import torch
import math
import gc

def clear_vram():
    """Dọn dẹp VRAM để các bài test không bị ảnh hưởng lẫn nhau."""
    torch.cuda.empty_cache()
    gc.collect()

def benchmark_forward(name, model, inputs, num_warmup=3, num_runs=10):
    """Hàm đo lường thời gian chạy và dung lượng VRAM tiêu thụ."""
    clear_vram()
    model = model.cuda().eval()
    
    # Chuyển inputs sang GPU
    inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]
    
    try:
        with torch.no_grad():
            # Warmup (Khởi động GPU)
            for _ in range(num_warmup):
                _ = model(*inputs)
            torch.cuda.synchronize()
            
            # Reset thông số đo RAM
            torch.cuda.reset_peak_memory_stats()
            
            # Đo thời gian bằng CUDA Event
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(num_runs):
                _ = model(*inputs)
            end_event.record()
            torch.cuda.synchronize()
            
            # Tính toán kết quả
            time_ms = start_event.elapsed_time(end_event) / num_runs
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
            return f"Thời gian: {time_ms:>6.2f} ms | VRAM: {mem_mb:>7.2f} MB"
            
    except RuntimeError as e:
        if "OutOfMemory" in str(e) or "out of memory" in str(e):
            return "OOM (Tràn VRAM) 💥"
        else:
            return f"Lỗi: {str(e)}"
    finally:
        del inputs
        clear_vram()

