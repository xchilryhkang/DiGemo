"""
complexity.py

Tiện ích phân tích model DiGemo từ một checkpoint đã lưu.

Bố cục:
    1. Phần load checkpoint dùng chung (build lại model + dummy input).
    2. Các hàm thực hiện nhiệm vụ:
        - calculate_complexity(): params, GFLOPs, inference time, peak memory.
        - predict(): chạy forward và trả về nhãn dự đoán (tối giản).

Cách dùng nhanh:
    python complexity.py --checkpoint ./checkpoints/best_model_IEMOCAP_9161.pth
    python complexity.py --checkpoint <path> --seq_len 50 --batch_size 1 --runs 100 --warmup 20
"""

import os
import time
import argparse
import statistics

import torch

from model import DiGemo

try:
    from thop import profile
    _HAS_THOP = True
except Exception:  # thop chưa cài
    _HAS_THOP = False


# ---------------------------------------------------------------------------
# Cấu hình dataset: embedding_dims = [text, visual, audio]
# n_speakers theo logic trong model.py (n_classes in {4, 6} -> 2, else 9).
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    "IEMOCAP":   {"embedding_dims": [1024, 342, 1582], "n_classes": 6, "n_speakers": 2},
    "IEMOCAP4":  {"embedding_dims": [1024, 512, 100],  "n_classes": 4, "n_speakers": 2},
    "MELD":      {"embedding_dims": [1024, 342, 300],  "n_classes": 7, "n_speakers": 9},
    "CMUMOSEI7": {"embedding_dims": [1024, 35, 384],   "n_classes": 7, "n_speakers": 9},
}


# ===========================================================================
# 1. LOAD CHECKPOINT DÙNG CHUNG
# ===========================================================================
def _safe_torch_load(path, map_location):
    """torch.load tương thích nhiều phiên bản (checkpoint chứa argparse.Namespace)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # torch cũ không có tham số weights_only
        return torch.load(path, map_location=map_location)


def _strip_module_prefix(state_dict):
    """Bỏ tiền tố 'module.' nếu checkpoint được lưu từ DDP."""
    if any(k.startswith("module.") for k in state_dict):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def resolve_dataset_config(args):
    """Lấy embedding_dims / n_classes / n_speakers từ args.dataset."""
    dataset = getattr(args, "dataset", None)
    if dataset not in DATASET_CONFIG:
        raise ValueError(
            f"Dataset '{dataset}' chưa được hỗ trợ. "
            f"Các lựa chọn hợp lệ: {list(DATASET_CONFIG.keys())}"
        )
    cfg = DATASET_CONFIG[dataset]
    return cfg["embedding_dims"], cfg["n_classes"], cfg["n_speakers"]


def load_model_from_checkpoint(checkpoint_path, device=None):
    """
    Load checkpoint -> dựng lại DiGemo -> nạp trọng số -> eval().

    Trả về:
        model        : DiGemo đã ở chế độ eval, trên đúng device.
        args         : argparse.Namespace lưu trong checkpoint.
        embedding_dims, n_classes, n_speakers
        device       : torch.device đang dùng.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    checkpoint = _safe_torch_load(checkpoint_path, map_location=device)
    args = checkpoint["args"]

    # Đồng bộ cờ no_cuda với device thực tế để tránh lỗi .cuda() bên trong model.
    args.no_cuda = (device.type == "cpu")

    embedding_dims, n_classes, n_speakers = resolve_dataset_config(args)

    model = DiGemo(args, embedding_dims, n_classes).to(device)
    state_dict = _strip_module_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    model.eval()

    return model, args, embedding_dims, n_classes, n_speakers, device


def build_dummy_inputs(embedding_dims, n_speakers, seq_len=50, batch_size=1, device="cpu"):
    """
    Tạo input giả đúng định dạng forward của DiGemo:
        forward(feature_t, feature_v, feature_a, umask, qmask, dia_lengths)
        - feature_*: (L, B, D)
        - umask    : (L, B)
        - qmask    : (L, B, n_speakers)
        - dia_lengths: list[int] độ dài mỗi hội thoại trong batch
    """
    device = torch.device(device)
    t_feat = torch.randn(seq_len, batch_size, embedding_dims[0], device=device)
    v_feat = torch.randn(seq_len, batch_size, embedding_dims[1], device=device)
    a_feat = torch.randn(seq_len, batch_size, embedding_dims[2], device=device)

    umask = torch.ones(seq_len, batch_size, device=device)

    qmask = torch.zeros(seq_len, batch_size, n_speakers, device=device)
    qmask[:, :, 0] = 1.0  # gán toàn bộ là speaker 0

    dia_lengths = [seq_len] * batch_size

    return (t_feat, v_feat, a_feat, umask, qmask, dia_lengths)


# ===========================================================================
# 2. CÁC HÀM ĐO COMPLEXITY
# ===========================================================================
def count_parameters(model):
    """Đếm tổng số tham số và số tham số trainable."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_flops_params(model, inputs):
    """
    Dùng thop.profile để ước lượng FLOPs và params.
    Lưu ý: thop có thể không đếm được một số op của graph conv -> con số FLOPs
    mang tính tham khảo. Trả về (flops, params) hoặc (None, None) nếu thiếu thop.
    """
    if not _HAS_THOP:
        print("[!] Chưa cài 'thop' (pip install thop) -> bỏ qua FLOPs.")
        return None, None
    with torch.no_grad():
        flops, params = profile(model, inputs=inputs, verbose=False)
    return flops, params


@torch.no_grad()
def measure_inference_time(model, inputs, warmup=20, runs=100, device="cpu"):
    """
    Đo thời gian inference trung bình cho 1 forward pass.
    Trả về dict: mean_ms, std_ms, min_ms, max_ms, fps.
    """
    device = torch.device(device)
    is_cuda = device.type == "cuda"

    # Warmup
    for _ in range(warmup):
        model(*inputs)
    if is_cuda:
        torch.cuda.synchronize()

    timings = []
    for _ in range(runs):
        if is_cuda:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            model(*inputs)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))  # ms
        else:
            t0 = time.perf_counter()
            model(*inputs)
            timings.append((time.perf_counter() - t0) * 1000.0)  # ms

    mean_ms = statistics.mean(timings)
    std_ms = statistics.stdev(timings) if len(timings) > 1 else 0.0
    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min(timings),
        "max_ms": max(timings),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
    }


@torch.no_grad()
def measure_peak_memory(model, inputs, device="cpu"):
    """
    Đo peak GPU memory cho 1 forward pass (MB).
    Chỉ có ý nghĩa trên CUDA; trên CPU trả về None.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    model(*inputs)
    torch.cuda.synchronize()

    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return {"peak_allocated_mb": peak_alloc, "peak_reserved_mb": peak_reserved}


def calculate_complexity(checkpoint_path, seq_len=50, batch_size=1,
                         warmup=20, runs=100, device=None, verbose=True):
    """
    NHIỆM VỤ CHÍNH: tính complexity của model từ checkpoint.
    Bao gồm: params, GFLOPs, inference time, peak memory.

    Trả về dict kết quả để có thể tái sử dụng/log.
    """
    model, args, embedding_dims, n_classes, n_speakers, device = \
        load_model_from_checkpoint(checkpoint_path, device)

    inputs = build_dummy_inputs(embedding_dims, n_speakers,
                                seq_len=seq_len, batch_size=batch_size,
                                device=device)

    # --- Params ---
    total_params, trainable_params = count_parameters(model)

    # --- FLOPs (thop) ---
    flops, thop_params = compute_flops_params(model, inputs)

    # --- Inference time ---
    timing = measure_inference_time(model, inputs, warmup=warmup, runs=runs, device=device)

    # --- Peak memory ---
    mem = measure_peak_memory(model, inputs, device=device)

    result = {
        "checkpoint": checkpoint_path,
        "dataset": getattr(args, "dataset", "unknown"),
        "device": str(device),
        "seq_len": seq_len,
        "batch_size": batch_size,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "flops": flops,
        "gflops": (flops / 1e9) if flops is not None else None,
        "timing": timing,
        "memory": mem,
    }

    if verbose:
        _print_report(result)

    return result


def _print_report(r):
    print("=" * 60)
    print(" DiGemo - Model Complexity Report")
    print("=" * 60)
    print(f" Checkpoint   : {r['checkpoint']}")
    print(f" Dataset      : {r['dataset']}")
    print(f" Device       : {r['device']}")
    print(f" Input shape  : seq_len={r['seq_len']}, batch_size={r['batch_size']}")
    print("-" * 60)
    print(f" Params (total)     : {r['total_params'] / 1e6:.3f} M")
    print(f" Params (trainable) : {r['trainable_params'] / 1e6:.3f} M")
    if r["gflops"] is not None:
        print(f" FLOPs              : {r['gflops']:.3f} GFLOPs")
    else:
        print(f" FLOPs              : N/A (cài 'thop' để tính)")
    t = r["timing"]
    print(f" Inference time     : {t['mean_ms']:.3f} ± {t['std_ms']:.3f} ms "
          f"(min {t['min_ms']:.3f} / max {t['max_ms']:.3f})")
    print(f" Throughput         : {t['fps']:.2f} forward/s")
    if r["memory"] is not None:
        m = r["memory"]
        print(f" Peak memory (alloc): {m['peak_allocated_mb']:.2f} MB")
        print(f" Peak memory (resv) : {m['peak_reserved_mb']:.2f} MB")
    else:
        print(f" Peak memory        : N/A (chỉ đo được trên CUDA)")
    print("=" * 60)


# ===========================================================================
# 3. PREDICT (tối giản) - chạy forward và trả về nhãn dự đoán
# ===========================================================================
@torch.no_grad()
def predict(checkpoint_path, inputs=None, seq_len=50, batch_size=1, device=None):
    """
    Load checkpoint rồi chạy dự đoán.
    - Nếu 'inputs' = None -> dùng dummy input (chủ yếu để smoke-test pipeline).
    - inputs phải đúng định dạng: (feature_t, feature_v, feature_a, umask, qmask, dia_lengths)

    Trả về (pred_labels, fused_logit) với pred_labels là tensor nhãn cảm xúc/utterance.
    """
    model, args, embedding_dims, n_classes, n_speakers, device = \
        load_model_from_checkpoint(checkpoint_path, device)

    if inputs is None:
        inputs = build_dummy_inputs(embedding_dims, n_speakers,
                                    seq_len=seq_len, batch_size=batch_size,
                                    device=device)
    else:
        # Đưa các tensor về đúng device
        moved = []
        for x in inputs:
            moved.append(x.to(device) if torch.is_tensor(x) else x)
        inputs = tuple(moved)

    fused_logit, t_logit, v_logit, a_logit, fused_feature = model(*inputs)
    pred_labels = torch.argmax(fused_logit, dim=-1)
    return pred_labels, fused_logit


# ===========================================================================
# CLI
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="DiGemo complexity & predict utility")
    p.add_argument("--checkpoint", type=str,
                   default="./checkpoints/best_model_IEMOCAP_9161.pth",
                   help="Đường dẫn tới file checkpoint .pth")
    p.add_argument("--seq_len", type=int, default=50, help="Độ dài chuỗi giả lập")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size giả lập")
    p.add_argument("--warmup", type=int, default=20, help="Số lần warmup khi đo thời gian")
    p.add_argument("--runs", type=int, default=100, help="Số lần đo thời gian inference")
    p.add_argument("--cpu", action="store_true", help="Ép chạy trên CPU")
    p.add_argument("--task", choices=["complexity", "predict"], default="complexity",
                   help="Nhiệm vụ cần chạy")
    return p.parse_args()


if __name__ == "__main__":
    cli = parse_args()
    dev = "cpu" if cli.cpu else None

    if cli.task == "complexity":
        calculate_complexity(
            cli.checkpoint,
            seq_len=cli.seq_len,
            batch_size=cli.batch_size,
            warmup=cli.warmup,
            runs=cli.runs,
            device=dev,
        )
    elif cli.task == "predict":
        preds, logits = predict(
            cli.checkpoint,
            seq_len=cli.seq_len,
            batch_size=cli.batch_size,
            device=dev,
        )
        print(f"Predicted labels shape: {tuple(preds.shape)}")
        print(f"Predicted labels: {preds.cpu().tolist()}")