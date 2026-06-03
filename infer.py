# by xueqianyue
import argparse
import csv
import json
import os
import sys
import time


CLASS_NAMES = ["lfm", "sfm", "bpsk", "fsk", "costas", "qpfm", "nlfm"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
CSV_EXTS = (".csv",)
INPUT_EXTS = IMAGE_EXTS + CSV_EXTS


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def import_runtime():
    global np, torch, T, Image, ImageDraw, ImageFont, autocast, scipy_signal, YAMLConfig

    import numpy as np
    import torch
    import torchvision.transforms as T
    from PIL import Image, ImageDraw, ImageFont
    from scipy import signal as scipy_signal
    from torch.cuda.amp import autocast

    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    from src.core import YAMLConfig


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA was requested but is unavailable; using CPU.")
        return torch.device("cpu")
    return device


def collect_inputs(input_path):
    if os.path.isfile(input_path):
        if os.path.splitext(input_path)[1].lower() not in INPUT_EXTS:
            raise ValueError(f"Unsupported input file: {input_path}")
        return [input_path]

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    paths = []
    for name in os.listdir(input_path):
        path = os.path.join(input_path, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in INPUT_EXTS:
            paths.append(path)
    return sorted(paths)


# by xueqianyue
def read_iq_csv(path):
    data = np.genfromtxt(path, delimiter=",", dtype=np.float64, invalid_raise=False)
    if data.size == 0:
        raise ValueError(f"Empty CSV file: {path}")
    if data.ndim == 1:
        if data.size <= 3:
            data = data.reshape(1, -1)
        else:
            data = data.reshape(-1, 1)
    data = data[np.all(np.isfinite(data), axis=1)]
    if data.size == 0:
        raise ValueError(f"No numeric rows found in CSV file: {path}")

    if data.shape[1] >= 3:
        times = data[:, 0].astype(np.float64)
        values = data[:, 1].astype(np.float64) + 1j * data[:, 2].astype(np.float64)
    elif data.shape[1] == 2:
        times = None
        values = data[:, 0].astype(np.float64) + 1j * data[:, 1].astype(np.float64)
    elif data.shape[1] == 1:
        times = None
        values = data[:, 0].astype(np.float64).astype(np.complex128)
    else:
        raise ValueError(f"CSV file must contain I/Q data: {path}")

    if values.size < 2:
        raise ValueError(f"CSV file is too short: {path}")
    return times, values.astype(np.complex128)


# by xueqianyue
def infer_sample_rate(times, default_fs):
    if times is None or len(times) < 2:
        return float(default_fs)
    diffs = np.diff(times.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return float(default_fs)
    dt = float(np.median(diffs))
    if dt <= 0:
        return float(default_fs)
    return 1.0 / dt


# by xueqianyue
def next_pow2(value):
    value = max(float(value), 1.0)
    return 1 << int(np.ceil(np.log2(value)))


# by xueqianyue
def mat2gray(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float64)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmax <= vmin:
        return np.zeros(values.shape, dtype=np.float64)
    return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)


# by xueqianyue
def fit_column_count(values, times, target_cols):
    target_cols = int(max(1, target_cols))
    if values.shape[1] == target_cols or times.size < 2:
        return values, times
    target_times = np.linspace(float(times[0]), float(times[-1]), target_cols)
    fitted = np.empty((values.shape[0], target_cols), dtype=values.dtype)
    for row_index in range(values.shape[0]):
        row = values[row_index]
        if np.iscomplexobj(row):
            fitted[row_index] = np.interp(target_times, times, row.real) + 1j * np.interp(
                target_times, times, row.imag
            )
        else:
            fitted[row_index] = np.interp(target_times, times, row)
    return fitted, target_times


# by xueqianyue
def compute_lam_spectrogram(values, fs, target_cols, args):
    nfft = next_pow2(2 * (int(args.stft_height) - 1))
    hop = max(1, int(np.floor(len(values) / max(int(target_cols), 1))))
    window_size = max(1, int(np.ceil(hop * float(args.preferred_window))))
    window_size = min(window_size, len(values))
    overlap = max(0, min(window_size - hop, window_size - 1))
    win = scipy_signal.windows.hamming(window_size, sym=False)
    _, spec_times, spec = scipy_signal.spectrogram(
        values,
        fs=float(fs),
        window=win,
        nperseg=window_size,
        noverlap=overlap,
        nfft=nfft,
        detrend=False,
        return_onesided=False,
        mode=args.stft_mode,
    )
    if spec.ndim != 2 or spec.shape[1] == 0:
        raise ValueError("Failed to build STFT from CSV input.")
    return fit_column_count(spec, spec_times, target_cols)


# by xueqianyue
def build_lam_image(segment, args):
    magnitude = np.abs(segment)
    max_power = float(np.max(magnitude)) if magnitude.size else 0.0
    if max_power <= 0 or not np.isfinite(max_power):
        lam_values = np.zeros_like(magnitude, dtype=np.float64)
    else:
        lam_values = float(args.lam_lambda) * np.log10(float(args.lam_epsilon) + magnitude / max_power)
    gray = np.rint(mat2gray(lam_values) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(gray).convert("RGB")


# by xueqianyue
def csv_duration_seconds(times, values, fs):
    sample_duration = len(values) / max(float(fs), 1.0)
    if times is None or len(times) < 2:
        return sample_duration
    time_duration = float(np.nanmax(times) - np.nanmin(times))
    if not np.isfinite(time_duration) or time_duration <= 0:
        return sample_duration
    return max(sample_duration, time_duration)


# by xueqianyue
def csv_to_images(path, args):
    times, values = read_iq_csv(path)
    fs = infer_sample_rate(times, args.fs)
    duration = csv_duration_seconds(times, values, fs)
    cut_time = float(args.cut_time)
    stem = os.path.splitext(os.path.basename(path))[0]

    use_long = args.csv_mode == "long" or (args.csv_mode == "auto" and duration > cut_time * 1.5)
    if not use_long:
        spec, _ = compute_lam_spectrogram(values, fs, int(args.stft_width), args)
        return [(f"{stem}.csv", build_lam_image(spec, args))]

    total_time = float(args.total_time) if float(args.total_time) > 0 else duration
    num_segments = max(1, int(np.floor(total_time / cut_time + 1e-9)))
    total_time = cut_time * num_segments if float(args.total_time) <= 0 else total_time
    target_cols = int(round(float(args.stft_width) * (total_time / cut_time)))
    spec, spec_times = compute_lam_spectrogram(values, fs, target_cols, args)
    images = []
    for index in range(num_segments):
        start = index * cut_time
        stop = (index + 1) * cut_time
        mask = (spec_times >= start) & (spec_times < stop)
        if not np.any(mask):
            continue
        segment = spec[:, mask]
        images.append((f"{stem}_{index + 1}.csv", build_lam_image(segment, args)))
    if not images:
        images.append((f"{stem}.csv", build_lam_image(spec, args)))
    return images


def iou_xyxy(a, b):
    x1, y1, x2, y2 = a
    xx1, yy1, xx2, yy2 = b
    ix1 = max(float(x1), float(xx1))
    iy1 = max(float(y1), float(yy1))
    ix2 = min(float(x2), float(xx2))
    iy2 = min(float(y2), float(yy2))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
    area_b = max(0.0, float(xx2) - float(xx1)) * max(0.0, float(yy2) - float(yy1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_same_label_overlaps(labels, boxes, scores, iou_thr):
    if len(boxes) == 0 or iou_thr <= 0:
        return labels, boxes, scores, list(range(len(boxes)))

    order = np.argsort(-scores)
    used = np.zeros(len(boxes), dtype=bool)
    keep_labels, keep_boxes, keep_scores, keep_sources = [], [], [], []

    for idx in order:
        if used[idx]:
            continue
        label = int(labels[idx])
        group = [int(idx)]
        used[idx] = True

        for j in order:
            if used[j] or int(labels[j]) != label:
                continue
            if iou_xyxy(boxes[idx], boxes[j]) >= iou_thr:
                used[j] = True
                group.append(int(j))

        xs, ys = [], []
        for gi in group:
            xs.extend([float(boxes[gi][0]), float(boxes[gi][2])])
            ys.extend([float(boxes[gi][1]), float(boxes[gi][3])])

        best_src = max(group, key=lambda gi: float(scores[gi]))
        keep_labels.append(label)
        keep_boxes.append([min(xs), min(ys), max(xs), max(ys)])
        keep_scores.append(float(scores[best_src]))
        keep_sources.append(best_src)

    return (
        np.asarray(keep_labels, dtype=np.int32),
        np.asarray(keep_boxes, dtype=np.float32),
        np.asarray(keep_scores, dtype=np.float32),
        keep_sources,
    )


class OnlinePrototypeBank:
    def __init__(self, num_classes, use_embeds, sim_thres, ema_m, w_yc, w_h, w_xw, w_cls):
        self.num_classes = int(num_classes)
        self.use_embeds = bool(use_embeds)
        self.sim_thres = float(sim_thres)
        self.ema_m = float(ema_m)
        self.w_yc = float(w_yc)
        self.w_h = float(w_h)
        self.w_xw = float(w_xw)
        self.w_cls = float(w_cls)
        self.items = []
        self.next_id = 1

    @staticmethod
    def l2n(vec):
        return vec / (np.linalg.norm(vec) + 1e-12)

    def make_feature(self, label, y_center_norm, h_norm, x_width_norm, embed):
        axis = np.asarray(
            [y_center_norm * self.w_yc, h_norm * self.w_h, x_width_norm * self.w_xw],
            dtype=np.float32,
        )
        onehot = np.zeros((self.num_classes,), dtype=np.float32)
        if 0 <= int(label) < self.num_classes:
            onehot[int(label)] = self.w_cls

        parts = [axis, onehot]
        if self.use_embeds:
            emb = np.zeros((128,), dtype=np.float32) if embed is None else np.asarray(embed, dtype=np.float32).reshape(-1)
            if emb.size != 128:
                emb = np.zeros((128,), dtype=np.float32)
            parts.append(self.l2n(emb))

        return self.l2n(np.concatenate(parts, axis=0))

    def assign(self, label, y_center_norm, h_norm, x_width_norm, embed):
        feat = self.make_feature(label, y_center_norm, h_norm, x_width_norm, embed)
        best_idx, best_sim = -1, -1.0

        for i, item in enumerate(self.items):
            sim = float(np.dot(feat, item["proto"]))
            if sim > best_sim:
                best_idx, best_sim = i, sim

        if best_idx >= 0 and best_sim >= self.sim_thres:
            item = self.items[best_idx]
            item["proto"] = self.l2n(self.ema_m * item["proto"] + (1.0 - self.ema_m) * feat)
            item["count"] += 1
            return int(item["id"]), best_sim

        cluster_id = self.next_id
        self.next_id += 1
        self.items.append({"id": cluster_id, "label": int(label), "proto": feat, "count": 1})
        return int(cluster_id), 1.0


def unpack_outputs(outputs):
    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 4:
            labels, boxes, scores, embeds = outputs
            return labels[0], boxes[0], scores[0], embeds[0]
        if len(outputs) == 3:
            labels, boxes, scores = outputs
            return labels[0], boxes[0], scores[0], None

    if isinstance(outputs, dict):
        outputs = [outputs]
    item = outputs[0]
    return (
        item.get("labels"),
        item.get("boxes"),
        item.get("scores"),
        item.get("embeds", item.get("embeddings")),
    )


def build_model(config_path, checkpoint_path, device):
    cfg = YAMLConfig(config_path, resume=checkpoint_path, PResNet={"pretrained": False})
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    missing, unexpected = cfg.model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")

    class Wrapper(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        @torch.no_grad()
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    return Wrapper(cfg).to(device).eval()


def tensor_to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def run_one_pil(image, display_name, frame_index, model, transform, bank, device, args):
    image = image.convert("RGB")
    width, height = image.size
    tensor = transform(image).unsqueeze(0).to(device)
    orig_size = torch.tensor([[width, height]], dtype=torch.float32, device=device)

    amp_on = bool(args.amp and device.type == "cuda")
    with torch.no_grad():
        with autocast(enabled=amp_on):
            outputs = model(tensor, orig_size)

    labels_t, boxes_t, scores_t, embeds_t = unpack_outputs(outputs)
    labels = tensor_to_numpy(labels_t).astype(np.int32).reshape(-1)
    boxes = tensor_to_numpy(boxes_t).astype(np.float32).reshape(-1, 4)
    scores = tensor_to_numpy(scores_t).astype(np.float32).reshape(-1)
    embeds = tensor_to_numpy(embeds_t)

    keep = scores >= float(args.conf_thres)
    labels, boxes, scores = labels[keep], boxes[keep], scores[keep]
    embeds = embeds[keep] if embeds is not None else None

    labels, boxes, scores, sources = merge_same_label_overlaps(labels, boxes, scores, args.iou_merge)
    embeds = embeds[sources] if embeds is not None and len(sources) > 0 else None

    if len(boxes) > 0 and args.boundary_margin > 0:
        margin = float(args.boundary_margin)
        inside = (
            (boxes[:, 0] >= margin)
            & (boxes[:, 1] >= margin)
            & (boxes[:, 2] <= width - margin)
            & (boxes[:, 3] <= height - margin)
        )
        labels, boxes, scores = labels[inside], boxes[inside], scores[inside]
        embeds = embeds[inside] if embeds is not None else None

    records = []
    for i in range(len(labels)):
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        label_id = int(labels[i])
        embed = None
        if args.use_embeds and embeds is not None:
            embed = np.asarray(embeds[i], dtype=np.float32).reshape(-1)

        cluster_id, similarity = bank.assign(
            label_id,
            ((y1 + y2) * 0.5) / max(float(height), 1.0),
            max(0.0, y2 - y1) / max(float(height), 1.0),
            max(0.0, x2 - x1) / max(float(width), 1.0),
            embed,
        )
        records.append(
            {
                "file": display_name,
                "frame_index": int(frame_index),
                "label_id": label_id,
                "label_name": CLASS_NAMES[label_id] if 0 <= label_id < len(CLASS_NAMES) else str(label_id),
                "score": float(scores[i]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cluster_id": int(cluster_id),
                "cluster_similarity": float(similarity),
            }
        )
    return image, records


def run_one_image(path, frame_index, model, transform, bank, device, args):
    image = Image.open(path).convert("RGB")
    return run_one_pil(image, os.path.basename(path), frame_index, model, transform, bank, device, args)


# by xueqianyue
def output_stem(display_name):
    stem = os.path.splitext(os.path.basename(display_name))[0]
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in stem)
    return cleaned or "input"


def draw_records(image, records, save_path):
    palette = [
        (214, 39, 40),
        (31, 119, 180),
        (44, 160, 44),
        (148, 103, 189),
        (255, 127, 14),
        (23, 190, 207),
        (188, 189, 34),
    ]
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for record in records:
        color = palette[int(record["label_id"]) % len(palette)]
        box = [record["x1"], record["y1"], record["x2"], record["y2"]]
        draw.rectangle(box, outline=color, width=3)
        text = f'{record["label_name"]} E{record["cluster_id"]} {record["score"]:.2f}'
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = max(0, int(record["x1"]))
        y = max(0, int(record["y1"]) - text_h - 6)
        draw.rectangle([x, y, x + text_w + 6, y + text_h + 6], fill=color)
        draw.text((x + 3, y + 3), text, fill=(255, 255, 255), font=font)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)


def write_outputs(records, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "predictions.csv")
    json_path = os.path.join(out_dir, "predictions.json")
    fields = [
        "file",
        "frame_index",
        "label_id",
        "label_name",
        "score",
        "x1",
        "y1",
        "x2",
        "y2",
        "cluster_id",
        "cluster_similarity",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return csv_path, json_path


def parse_args():
    parser = argparse.ArgumentParser("Uni-RSNet final inference")
    parser.add_argument("--input", "-i", required=True, help="csv, image, or folder")
    parser.add_argument("--config", "-c", default="configs/unirsnet.yml")
    parser.add_argument("--resume", "-r", default="weights/unirsnet_final.pth")
    parser.add_argument("--out-dir", "-o", default="outputs/infer")
    parser.add_argument("--device", "-d", default="auto")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--resize", type=int, default=1024)
    parser.add_argument("--conf-thres", type=float, default=0.80)
    parser.add_argument("--iou-merge", type=float, default=0.55)
    parser.add_argument("--boundary-margin", type=float, default=3.0)

    parser.add_argument("--use-embeds", type=str2bool, default=True)
    parser.add_argument("--sim-thres", type=float, default=0.99)
    parser.add_argument("--ema-m", type=float, default=0.95)
    parser.add_argument("--w-yc", type=float, default=10.0)
    parser.add_argument("--w-h", type=float, default=10.0)
    parser.add_argument("--w-xw", type=float, default=10.0)
    parser.add_argument("--w-cls", type=float, default=1.0)

    # by xueqianyue
    parser.add_argument("--fs", type=float, default=100e6)
    parser.add_argument("--cut-time", type=float, default=0.0002)
    parser.add_argument("--total-time", type=float, default=0.0)
    parser.add_argument("--stft-height", type=int, default=1024)
    parser.add_argument("--stft-width", type=int, default=2048)
    parser.add_argument("--preferred-window", type=float, default=20.0)
    parser.add_argument("--lam-lambda", "--lambda", dest="lam_lambda", type=float, default=10.0)
    parser.add_argument("--lam-epsilon", type=float, default=0.01)
    parser.add_argument("--csv-mode", choices=("auto", "segment", "long"), default="auto")
    parser.add_argument("--stft-mode", choices=("psd", "magnitude", "complex"), default="psd")
    parser.add_argument("--no-save-vis", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    import_runtime()

    if not os.path.exists(args.resume):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.resume}. Put the final weight at this path or pass --resume."
        )

    inputs = collect_inputs(args.input)
    if not inputs:
        raise FileNotFoundError(f"No supported inputs found under: {args.input}")

    device = resolve_device(args.device)
    resize_hw = (args.resize, args.resize) if args.resize > 0 else None
    transform = T.Compose([T.Resize(resize_hw) if resize_hw else (lambda x: x), T.ToTensor()])
    model = build_model(args.config, args.resume, device)
    bank = OnlinePrototypeBank(
        num_classes=args.num_classes,
        use_embeds=args.use_embeds,
        sim_thres=args.sim_thres,
        ema_m=args.ema_m,
        w_yc=args.w_yc,
        w_h=args.w_h,
        w_xw=args.w_xw,
        w_cls=args.w_cls,
    )

    all_records = []
    vis_dir = os.path.join(args.out_dir, "vis")
    t0 = time.time()
    frame_index = 0
    for path in inputs:
        ext = os.path.splitext(path)[1].lower()
        if ext in CSV_EXTS:
            prepared = csv_to_images(path, args)
        else:
            prepared = [(os.path.basename(path), Image.open(path).convert("RGB"))]
        for display_name, image in prepared:
            image, records = run_one_pil(image, display_name, frame_index, model, transform, bank, device, args)
            all_records.extend(records)
            if not args.no_save_vis:
                draw_records(image, records, os.path.join(vis_dir, f"{output_stem(display_name)}_pred.png"))
            frame_index += 1
            print(f"[{frame_index}] {display_name} detections={len(records)}")

    if frame_index == 0:
        raise RuntimeError(f"No inference frames were generated from: {args.input}")

    csv_path, json_path = write_outputs(all_records, args.out_dir)
    print(f"[done] inputs={frame_index} detections={len(all_records)} time={time.time() - t0:.2f}s")
    print(f"[csv] {csv_path}")
    print(f"[json] {json_path}")


if __name__ == "__main__":
    main()
