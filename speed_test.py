import torch
import math
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

class OptimizedSampledAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def select_important_tokens(self, tensor, topk):
        mean = tensor.mean(dim=-1)
        std = tensor.std(dim=-1)
        importance_score = mean + std
        _, top_indices = torch.topk(importance_score, topk, dim=-1)
        return top_indices
    
    @torch.compile(mode="reduce-overhead")
    def forward(self, q, k, v):
        B, num_heads, seq_len, d_k = q.shape
        sample_num = seq_len
        topk = int(2 * math.sqrt(sample_num))
        
        sample_idx = self.select_important_tokens(q, topk)
        expanded_indices = sample_idx.unsqueeze(-1).expand(-1, -1, -1, d_k).contiguous()
        
        q_sampled = torch.gather(q, 2, expanded_indices).contiguous()
        k_sampled = torch.gather(k, 2, expanded_indices).contiguous()
        v_sampled = torch.gather(v, 2, expanded_indices).contiguous()
        
        sqrt_dk = math.sqrt(d_k)
        attention_scores = torch.matmul(q_sampled, k_sampled.transpose(-2, -1)) / sqrt_dk
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        sampled_attention = torch.matmul(attention_weights, v_sampled)
        qk_result_expanded = torch.empty_like(v, device=v.device, dtype=v.dtype)
        qk_result_expanded.scatter_(2, expanded_indices, sampled_attention)
        
        H = W = int(math.sqrt(seq_len))
        sampled_result = qk_result_expanded.reshape(B, -1, H, W)
        return sampled_result

@torch.compile(mode="reduce-overhead")
def classic_attention(q, k, v):
    B, num_heads, seq_len, d_k = q.shape
    
    def kernel(x):
        return F.elu(x) + 1.0
    
    q_kernel = kernel(q)
    k_kernel = kernel(k)
    
    k_t = k_kernel.transpose(-2, -1)
    k_t_v = torch.matmul(k_t, v)
    q_kv = torch.matmul(q_kernel, k_t_v)
    
    k_sum = k_kernel.sum(dim=-2, keepdim=True)
    norm_factor = torch.matmul(q_kernel, k_sum.transpose(-2, -1)) + 1e-6
    attention_output = q_kv / norm_factor
    
    H = W = int(math.sqrt(seq_len))
    linear_result = attention_output.reshape(B, -1, H, W)
    return linear_result

class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    @torch.compile(mode="reduce-overhead")
    def forward(self, x):
        result_conv = self.conv3x3(x)
        result_conv = self.conv5x5(result_conv)
        ca = self.channel_attention(result_conv)
        return result_conv * ca

@torch.compile(mode="reduce-overhead")
def focused_kernel_attention(q, k, v, eps=1e-6):
    B, num_heads, seq_len, d_k = q.shape
    H = W = int(math.sqrt(seq_len))
    
    q_norm = q.norm(dim=-2, keepdim=True)
    k_norm = k.norm(dim=-2, keepdim=True)
    
    q = q ** 3
    k = k ** 3
    
    q = (q / q.norm(dim=-2, keepdim=True)) * q_norm
    k = (k / k.norm(dim=-2, keepdim=True)) * k_norm
    
    trans_k = k.transpose(-1, -2)
    v = F.pad(v, (0, 1), mode="constant", value=1)
    kv = torch.matmul(trans_k, v)
    out = torch.matmul(q, kv)
    out = out[..., :-1] / (out[..., -1:] + eps)
    out = torch.transpose(out, -1, -2)
    return out.reshape(B, -1, H, W)

class TokenInteraction(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dwc = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=5,
            groups=in_channels, 
            padding=5 // 2
        )
    @torch.compile(mode="reduce-overhead")
    def forward(self, v_reshaped):
        return self.dwc(v_reshaped)

class FeatureAdaptive(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.weight_generator = nn.Linear(in_channels, 3)
    @torch.compile(mode="reduce-overhead")
    def forward(self, q_reshaped):
        x = self.global_pool(q_reshaped)
        x = x.flatten(1)
        return self.weight_generator(x)

def measure_average_time(func, data_list, use_amp=True):
    for data in data_list[:10]:
        if use_amp and torch.cuda.is_available():
            with autocast():
                _ = func(*data)
        else:
            _ = func(*data)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    
    for data in data_list:
        if use_amp and torch.cuda.is_available():
            with autocast():
                _ = func(*data)
        else:
            _ = func(*data)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    
    total_time = end - start
    avg_time = total_time / len(data_list)
    return avg_time, total_time

if __name__ == "__main__":
    BATCH_SIZE = 8
    NUM_HEADS = 8        
    SEQ_LEN = 256        
    D_K = 64             
    NUM_TESTS = 1000     
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (torch.cuda.is_available() and device == "cuda") else torch.float32
    
    H = W = int(math.sqrt(SEQ_LEN))
    in_channels = NUM_HEADS * D_K
    
    test_data_original = []  
    test_data_ca = []        
    test_data_fk = []        
    test_data_ti = []        
    test_data_fa = []        
    
    for _ in range(NUM_TESTS):
        q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_K, device=device, dtype=dtype)
        k = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_K, device=device, dtype=dtype)
        v = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_K, device=device, dtype=dtype)
        
        x_ca = q.reshape(BATCH_SIZE, -1, H, W).contiguous()
        
        v_reshaped = v.reshape(BATCH_SIZE, -1, H, W).contiguous()

        q_reshaped = q.reshape(BATCH_SIZE, NUM_HEADS, H, W, D_K)\
                     .permute(0, 1, 4, 2, 3)\
                     .reshape(BATCH_SIZE, NUM_HEADS*D_K, H, W)\
                     .contiguous()
        
        test_data_original.append((q, k, v))
        test_data_ca.append((x_ca,))
        test_data_fk.append((q, k, v))
        test_data_ti.append((v_reshaped,))
        test_data_fa.append((q_reshaped,))
    
    optimized_sampled_attn = OptimizedSampledAttention().to(device, dtype=dtype)
    channel_attn = ChannelAttentionBlock(in_channels).to(device, dtype=dtype)
    token_interact = TokenInteraction(in_channels).to(device, dtype=dtype)
    feature_adapt = FeatureAdaptive(in_channels).to(device, dtype=dtype)
    
    def optimized_sampled_wrapper(q, k, v):
        return optimized_sampled_attn(q, k, v)
    
    def channel_attn_wrapper(x_ca):
        return channel_attn(x_ca)
    
    def focused_kernel_wrapper(q, k, v):
        return focused_kernel_attention(q, k, v)
    
    def token_interact_wrapper(v_reshaped):
        return token_interact(v_reshaped)
    
    def feature_adapt_wrapper(q_reshaped):
        return feature_adapt(q_reshaped)
    
    sampled_avg, sampled_total = measure_average_time(optimized_sampled_wrapper, test_data_original)
    linear_avg, linear_total = measure_average_time(classic_attention, test_data_original)
    ca_avg, ca_total = measure_average_time(channel_attn_wrapper, test_data_ca)
    fk_avg, fk_total = measure_average_time(focused_kernel_wrapper, test_data_fk)
    ti_avg, ti_total = measure_average_time(token_interact_wrapper, test_data_ti)
    fa_avg, fa_total = measure_average_time(feature_adapt_wrapper, test_data_fa)
    
    print("="*70)
    print(f"Test Environment：{device.upper()} | Batch Size：{BATCH_SIZE} | Seq Len：{SEQ_LEN}")
    print("-"*70)
    print(f"Selective Sampling SoftMax：")
    print(f"  Total：{sampled_total:.6f} s | Single-sample Average：{sampled_avg*1000:.6f} ms")
    print("-"*70)
    print(f"Standard Linear Attention：")
    print(f"  Total：{linear_total:.6f} s | Single-sample Average：{linear_avg*1000:.6f} ms")
    print("-"*70)
    print(f"Channel Attention：")
    print(f"  Total：{ca_total:.6f} s | Single-sample Average：{ca_avg*1000:.6f} ms")
    print("-"*70)
    print(f"Focused Linear Attention：")
    print(f"  Total：{fk_total:.6f} s | Single-sample Average：{fk_avg*1000:.6f} ms")
    print("-"*70)
    print(f"Token Interaction：")
    print(f"  Total：{ti_total:.6f} s | Single-sample Average：{ti_avg*1000:.6f} ms")
    print("-"*70)
    print(f"Feature Adaptive Fusion：")
    print(f"  Total：{fa_total:.6f} s | Single-sample Average：{fa_avg*1000:.6f} ms")
    print("="*70)
