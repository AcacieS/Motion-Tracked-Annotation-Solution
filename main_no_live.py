import os, cv2, tempfile
import numpy as np
from PIL import Image
import torch
import gradio as gr
from omegaconf import open_dict
import time


OUT_DIR = os.path.join(tempfile.gettempdir(), "cutie_gradio")
os.makedirs(OUT_DIR, exist_ok=True)

import traceback
from cutie.inference.inference_core import InferenceCore
from gui.interactive_utils import image_to_torch, torch_prob_to_numpy_mask, index_numpy_to_one_hot_torch, overlay_davis

DEFAULT_VIDEO = "Lapchole4.mp4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# def _resolve_video_path(p):
#     #cand = [p, os.path.join("/content", p)]
#     cand = [p, os.path.join(os.getcwd(), p)]
#     for c in cand:
#         if os.path.exists(c):
#             return c
#     raise gr.Error(f"Video not found: {p}")

def _get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30.0 if (fps is None or fps <= 1e-3) else float(fps)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return fps, n, w, h

def resolve_video_path(video_input):
    # Gradio older versions return dict {name: "..."}
    if isinstance(video_input, dict):
        return video_input["name"]
    return video_input

def _read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if (not ok) or frame is None:
        raise gr.Error(f"Failed to read frame {frame_idx}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, Image.fromarray(frame_rgb)

def _editor_value_from_frame(frame_pil):
    # 关键：让你“在这一帧上画”，不是黑底
    # ImageEditor 需要 composite 字段，否则你之前会 KeyError
    return {"background": frame_pil, "layers": [], "composite": frame_pil}

def _warp_mask_affine(mask01, M, w, h):
    """
    Warp binary mask with affine transform.
    mask01 : HxW uint8 (0/1)
    M      : 2x3 affine matrix mapping prev->curr
    w,h    : output size
    returns: warped mask HxW uint8 (0/1)
    """
    import cv2
    import numpy as np
    
    mask_u8 = (mask01 > 0).astype(np.uint8) * 255
    warped = cv2.warpAffine(
        mask_u8,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return (warped > 127).astype(np.uint8)

def _mask_from_editor(editor_value):
    """
    从 ImageEditor 取 mask：用 composite 和 background 的像素差分得到前景区域
    你在帧上画的地方会改变 composite 像素 -> diff>阈值 -> mask=1
    """
    if editor_value is None:
        raise gr.Error("Mask editor is empty. Please paint on the frame.")
    bg = editor_value.get("background", None)
    comp = editor_value.get("composite", None) or bg
    if bg is None or comp is None:
        raise gr.Error("ImageEditor returned no background/composite.")

    bg = bg.convert("RGB")
    comp = comp.convert("RGB")
    bg_arr = np.array(bg).astype(np.int16)
    cp_arr = np.array(comp).astype(np.int16)

    if bg_arr.shape != cp_arr.shape:
        raise gr.Error("Editor output size mismatch. Try reloading the frame.")

    diff = np.abs(cp_arr - bg_arr).sum(axis=-1)  # H,W
    mask = (diff > 25).astype(np.uint8)          # 阈值可调：越大越不敏感
    return mask

def _save_overlay_video(frames_bgr, fps, out_mp4):
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))
    for f in frames_bgr:
        vw.write(f)
    vw.release()


def load_video(video_path_str):
    vp = resolve_video_path(video_path_str)
    # vp = video_path_str
    if vp is None:
        raise gr.Error("Please upload a video first.")
    fps, n, w, h = _get_video_info(vp)
    _, frame0_pil = _read_frame(vp, 0)
    editor_init = _editor_value_from_frame(frame0_pil)

    info = f"Loaded: {vp} | frames={n} | fps={fps:.2f} | size={w}x{h}"

    # 选帧 slider：0..n-1
    frame_idx_update = gr.update(minimum=0, maximum=max(0, n-1), value=0, step=1)

    # frames_to_propagate 上限：n-1-当前帧（当前帧=0）
    max_prop = max(1, (n - 1) - 0)
    frames_to_prop_update = gr.update(minimum=1, maximum=max_prop, value=min(200, max_prop), step=1)

    # max_internal_size：建议不超过 max(w,h)，越小越快越糊
    max_side = max(w, h)
    default_mis = min(max_side, 720)  # 你也可以改成 640/800 等
    max_internal_update = gr.update(minimum=256, maximum=max(256, max_side), value=default_mis, step=32)

    return vp, frame0_pil, editor_init, frame_idx_update, frames_to_prop_update, max_internal_update, info


def overlay_fast_bgr(frame_bgr, mask01, color_bgr=(0, 255, 0), alpha=0.45, draw_outline=True):
    """
    frame_bgr: HxWx3 uint8 (OpenCV BGR)
    mask01:    HxW, 0/1 uint8 (or bool)
    return:    HxWx3 uint8 BGR
    """
    if mask01 is None:
        return frame_bgr
    m = (mask01 > 0)
    if not m.any():
        return frame_bgr  # 没有前景就原样返回（不要报错/不要花时间）

    out = frame_bgr.copy()

    # 只对 mask 区域做 alpha blend（避免生成整张 color layer）
    # out[m] = (1-a)*frame + a*color
    a = float(alpha)
    inv = 1.0 - a
    c0, c1, c2 = map(float, color_bgr)  # B,G,R
    region = out[m].astype(np.float32)
    region[:, 0] = region[:, 0] * inv + c0 * a
    region[:, 1] = region[:, 1] * inv + c1 * a
    region[:, 2] = region[:, 2] * inv + c2 * a
    out[m] = region.astype(np.uint8)

    if draw_outline:
        # 画一圈轮廓（很便宜，但可视效果提升大）
        mu8 = (m.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(out, contours, -1, (0, 0, 255), 2)  # 红色边

    return out



def _estimate_affine_lk(prev_bgr, curr_bgr, prev_mask01,
                        max_corners=400,
                        quality=0.01,
                        min_dist=7,
                        ransac_thresh=3.0,
                        lk_scale=0.5,          # <<< 新增：LK 用的缩放比例(0<scale<=1)，0.5 很常用
                        win_size=21,
                        max_level=3):
    """
    低分辨率 LK + 仿射估计（返回原始分辨率坐标系的 M）

    prev_mask01: HxW, 0/1 uint8
    返回: (M, inlier_ratio, n_good)
      M: 2x3 仿射矩阵 (用于原尺寸 warpAffine) or None
    """
    import cv2
    import numpy as np

    # --- safety ---
    s = float(lk_scale)
    if not (0.0 < s <= 1.0):
        s = 0.5

    # --- grayscale ---
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

    h, w = prev_gray.shape[:2]

    # --- downscale for LK ---
    if s < 1.0:
        new_w = max(32, int(w * s))
        new_h = max(32, int(h * s))
        prev_s = cv2.resize(prev_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        curr_s = cv2.resize(curr_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # mask 缩放用最近邻，保持 0/1
        mask_u8 = (prev_mask01 > 0).astype(np.uint8) * 255
        mask_s = cv2.resize(mask_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        prev_s, curr_s = prev_gray, curr_gray
        mask_s = (prev_mask01 > 0).astype(np.uint8) * 255

    # --- scale params for small image (optional but helps stability) ---
    # min_dist / blockSize 在小图上也应该缩放
    min_dist_s = max(1, int(min_dist * s)) if s < 1.0 else int(min_dist)
    block_s = 7
    if s < 1.0:
        block_s = max(3, int(7 * s))
        if block_s % 2 == 0:
            block_s += 1

    # --- 1) find corners on small image (prefer mask region) ---
    pts0 = cv2.goodFeaturesToTrack(
        prev_s, maxCorners=max_corners,
        qualityLevel=quality, minDistance=min_dist_s,
        blockSize=block_s, mask=mask_s
    )
    # fallback to full small image (still cheap)
    if pts0 is None or len(pts0) < 8:
        pts0 = cv2.goodFeaturesToTrack(
            prev_s, maxCorners=max_corners,
            qualityLevel=quality, minDistance=min_dist_s,
            blockSize=block_s, mask=None
        )

    if pts0 is None or len(pts0) < 8:
        return None, 0.0, 0

    # --- 2) LK flow on small image ---
    pts1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_s, curr_s, pts0, None,
        winSize=(int(win_size), int(win_size)),
        maxLevel=int(max_level),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    st = st.reshape(-1)
    good0 = pts0.reshape(-1, 2)[st == 1]
    good1 = pts1.reshape(-1, 2)[st == 1]

    if good0.shape[0] < 8:
        return None, 0.0, int(good0.shape[0])

    # --- 3) map points back to original resolution coords ---
    if s < 1.0:
        inv = 1.0 / s
        good0 = good0 * inv
        good1 = good1 * inv

    # --- 4) affine with RANSAC (in original coords) ---
    M, inliers = cv2.estimateAffinePartial2D(
        good0, good1,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_thresh),
        maxIters=2000, confidence=0.99, refineIters=10
    )
    if M is None or inliers is None:
        return None, 0.0, int(good0.shape[0])

    inlier_ratio = float(inliers.sum()) / max(1, len(inliers))
    return M, inlier_ratio, int(good0.shape[0])





def show_frame(video_path_str, frame_idx):
    vp = resolve_video_path(video_path_str)
    # vp = video_path_str
    if vp is None:
        raise gr.Error("Please upload a video first.")
    fps, n, w, h = _get_video_info(vp)

    frame_idx = int(frame_idx)
    _, frame_pil = _read_frame(vp, frame_idx)

    # 切到该帧时，frames_to_propagate 上限跟着变：n-1-当前帧
    max_prop = max(1, (n - 1) - frame_idx)
    frames_to_prop_update = gr.update(minimum=1, maximum=max_prop, value=min(200, max_prop), step=1)

    return frame_pil, _editor_value_from_frame(frame_pil), frames_to_prop_update

def run_track(video_path_str, start_frame_idx, editor_value, frames_to_propagate, max_internal_size,
              lk_every, lk_corners, lk_inlier):
    """
    Anchor-based design (your idea) + ROI (added):
    - CUTIE runs ONLY on anchor frames: every K (=lk_every) frames.
      (Except the first frame uses user mask to initialize.)
    - CUTIE runs on a FIXED-SIZE ROI crop (square). ROI follows the object center estimated from masks.
      This keeps CUTIE input resolution smaller and *constant*, which is safer for InferenceCore's memory.
    - Between two CUTIE anchors, use LK affine warps:
        * forward warp from prev anchor to intermediate frames
        * backward warp from next anchor to intermediate frames
        * blend the two warped masks (linear weight by position)
    - LK is computed only in a window around the mask (big speed-up for LK).
    - Overlay is FAST OpenCV alpha-blend (+ optional contour), no overlay_davis.
    - Keeps full timing breakdown.
    """
    import time
    import numpy as np
    import cv2
    import torch
    import gradio as gr
    from omegaconf import open_dict

    # -------------------------
    # fast overlay (BGR)
    # -------------------------
    def overlay_fast_bgr(frame_bgr, mask01, color_bgr=(0, 255, 0), alpha=0.45, draw_outline=True):
        if mask01 is None:
            return frame_bgr
        m = (mask01 > 0)
        if not m.any():
            return frame_bgr

        out = frame_bgr.copy()
        a = float(alpha)
        inv = 1.0 - a
        c0, c1, c2 = map(float, color_bgr)  # B,G,R

        region = out[m].astype(np.float32)
        region[:, 0] = region[:, 0] * inv + c0 * a
        region[:, 1] = region[:, 1] * inv + c1 * a
        region[:, 2] = region[:, 2] * inv + c2 * a
        out[m] = region.astype(np.uint8)

        if draw_outline:
            mu8 = (m.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(out, contours, -1, (0, 0, 255), 2)  # red outline
        return out

    # -------------------------
    # parse params
    # -------------------------
    K = int(lk_every)
    lk_corners = int(lk_corners)
    lk_inlier = float(lk_inlier)
    if K < 1:
        K = 1

    vp = resolve_video_path(video_path_str)
    # vp = video_path_str
    if vp is None:
        raise gr.Error("Please upload a video first.")
    fps, n, w, h = _get_video_info(vp)

    start_frame_idx = int(start_frame_idx)
    frames_to_propagate = int(frames_to_propagate)
    max_internal_size = int(max_internal_size)

    remaining = (n - 1) - start_frame_idx
    if remaining <= 0:
        raise gr.Error(f"No remaining frames from start_frame={start_frame_idx}. video_frames={n}")
    frames_to_propagate = max(1, min(frames_to_propagate, remaining))

    # user mask (full-res)
    mask_index = _mask_from_editor(editor_value)
    if mask_index.sum() < 10:
        raise gr.Error("Mask too small / empty. Please paint a larger region on the frame.")
    if mask_index.shape[0] != h or mask_index.shape[1] != w:
        mask_index = cv2.resize(mask_index, (w, h), interpolation=cv2.INTER_NEAREST)

    # set cfg
    with open_dict(cfg):
        cfg["max_internal_size"] = max_internal_size

    processor = InferenceCore(cutie, cfg=cfg)

    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        raise gr.Error(f"Cannot open video: {vp}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

    overlay_frames_bgr = []

    # stats
    model_calls = 0
    lk_pairs = 0
    lk_bad = 0

    # timing
    t_total0 = time.perf_counter()
    t_read = 0.0
    t_model = 0.0
    t_lk = 0.0
    t_overlay = 0.0
    t_misc = 0.0
    t_save = 0.0
    t_roi = 0.0  # ROI crop/uncrop bookkeeping

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # -------------------------------------------------------
    # ROI helpers (FIXED SIZE square ROI)
    # -------------------------------------------------------
    # ROI side is derived from max_internal_size for convenience:
    # max_internal_size=320 => roi_side ~ 640 (you can tune this factor)
    ROI_FACTOR = 2.0
    roi_side = int(max(256, min(min(w, h), min(1024, round(max_internal_size * ROI_FACTOR)))))
    if roi_side > min(w, h):
        roi_side = min(w, h)

    def _bbox_from_mask(mask01):
        ys, xs = np.where(mask01 > 0)
        if ys.size == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        return x0, y0, x1, y1

    def _center_from_mask(mask01, fallback_cx, fallback_cy):
        bb = _bbox_from_mask(mask01)
        if bb is None:
            return fallback_cx, fallback_cy
        x0, y0, x1, y1 = bb
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        return cx, cy

    def _roi_from_center(cx, cy):
        # clamp ROI so it's always fully inside the frame (constant roi_side => constant CUTIE input size)
        x0 = int(round(cx - roi_side / 2))
        y0 = int(round(cy - roi_side / 2))
        x0 = max(0, min(x0, w - roi_side))
        y0 = max(0, min(y0, h - roi_side))
        return x0, y0, x0 + roi_side, y0 + roi_side

    def _crop_roi(frame_bgr, roi):
        x0, y0, x1, y1 = roi
        return frame_bgr[y0:y1, x0:x1]

    def _place_roi_mask_to_full(mask_roi01, roi):
        # mask_roi01: roi_side x roi_side
        x0, y0, x1, y1 = roi
        full = np.zeros((h, w), dtype=np.uint8)
        full[y0:y1, x0:x1] = mask_roi01.astype(np.uint8)
        return full

    # initial ROI center from user mask
    cx0, cy0 = _center_from_mask(mask_index, w / 2, h / 2)
    roi = _roi_from_center(cx0, cy0)

    # -------------------------------------------------------
    # helper: LK warp in a WINDOW around mask (big speed-up)
    # -------------------------------------------------------
    def _compose_affine_with_crop(M, x0, y0):
        # M maps (prev_crop) -> (curr_crop)
        # full mapping: p_full -> p_full':
        # p' = M*(p - t) + t, where t=(x0,y0)
        a, b, c = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
        d, e, f = float(M[1, 0]), float(M[1, 1]), float(M[1, 2])
        tx, ty = float(x0), float(y0)
        c2 = c + tx - a * tx - b * ty
        f2 = f + ty - d * tx - e * ty
        return np.array([[a, b, c2], [d, e, f2]], dtype=np.float32)

    def _estimate_affine_lk_safe(prev_bgr, curr_bgr, prev_mask01):
        # you may have already upgraded _estimate_affine_lk signature;
        # this wrapper won't crash if it's still the old one.
        try:
            return _estimate_affine_lk(
                prev_bgr, curr_bgr, prev_mask01,
                max_corners=lk_corners,
                lk_scale=0.33,
                win_size=21,
                max_level=3
            )
        except TypeError:
            return _estimate_affine_lk(
                prev_bgr, curr_bgr, prev_mask01,
                max_corners=lk_corners
            )

    def warp_mask_lk(prev_frame_bgr, curr_frame_bgr, prev_mask01):
        nonlocal lk_pairs, lk_bad, t_lk, t_misc

        # if mask empty => nothing to track
        bb = _bbox_from_mask(prev_mask01)
        if bb is None:
            lk_bad += 1
            return prev_mask01

        # window around mask bbox
        x0, y0, x1, y1 = bb

        # pad window (tune this)
        PAD = 64
        wx0 = max(0, x0 - PAD)
        wy0 = max(0, y0 - PAD)
        wx1 = min(w, x1 + PAD)
        wy1 = min(h, y1 + PAD)

        # ensure window has some size
        if (wx1 - wx0) < 64 or (wy1 - wy0) < 64:
            # expand minimally
            wx0 = max(0, int((x0 + x1) / 2 - 64))
            wy0 = max(0, int((y0 + y1) / 2 - 64))
            wx1 = min(w, wx0 + 128)
            wy1 = min(h, wy0 + 128)

        prev_crop = prev_frame_bgr[wy0:wy1, wx0:wx1]
        curr_crop = curr_frame_bgr[wy0:wy1, wx0:wx1]
        mask_crop = prev_mask01[wy0:wy1, wx0:wx1]

        t0 = time.perf_counter()
        M, inlier_ratio, n_good = _estimate_affine_lk_safe(prev_crop, curr_crop, mask_crop)
        t_lk += (time.perf_counter() - t0)
        lk_pairs += 1

        if (M is None) or (inlier_ratio < lk_inlier) or (n_good < 8):
            lk_bad += 1
            return prev_mask01

        t0 = time.perf_counter()
        M_full = _compose_affine_with_crop(M, wx0, wy0)
        out = _warp_mask_affine(prev_mask01, M_full, w, h).astype(np.uint8)
        t_misc += (time.perf_counter() - t0)
        return out

    # -------------------------------------------------------
    # 1) read first frame + init CUTIE with user mask (ON ROI)
    # -------------------------------------------------------
    processed = 0

    t0 = time.perf_counter()
    ok, frame0 = cap.read()
    t_read += (time.perf_counter() - t0)
    if (not ok) or frame0 is None:
        cap.release()
        raise gr.Error("Failed to read the start frame.")

    # crop ROI for CUTIE init
    t0 = time.perf_counter()
    frame0_roi = _crop_roi(frame0, roi)
    mask0_roi = _crop_roi(mask_index, roi)
    t_roi += (time.perf_counter() - t0)

    with torch.no_grad():
        amp_device = "cuda" if DEVICE == "cuda" else "cpu"
        with torch.amp.autocast(device_type=amp_device, enabled=(DEVICE == "cuda")):
            t0 = time.perf_counter()
            frame0_t = image_to_torch(frame0_roi, device=DEVICE)
            mask_torch = index_numpy_to_one_hot_torch(mask0_roi, 2).to(DEVICE)
            pred0 = processor.step(frame0_t, mask_torch[1:], idx_mask=False)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t_model += (time.perf_counter() - t0)
            model_calls += 1

    t0 = time.perf_counter()
    anchor_mask_roi = torch_prob_to_numpy_mask(pred0).astype(np.uint8)  # roi_side x roi_side
    anchor_mask = _place_roi_mask_to_full(anchor_mask_roi, roi)         # full HxW
    anchor_frame = frame0
    t_misc += (time.perf_counter() - t0)

    # update ROI center based on anchor_mask (keep ROI following)
    t0 = time.perf_counter()
    cx, cy = _center_from_mask(anchor_mask, cx0, cy0)
    roi = _roi_from_center(cx, cy)
    t_roi += (time.perf_counter() - t0)

    t0 = time.perf_counter()
    overlay_frames_bgr.append(
        overlay_fast_bgr(anchor_frame, anchor_mask, alpha=0.45, draw_outline=True)
    )
    t_overlay += (time.perf_counter() - t0)
    processed += 1

    # -------------------------------------------------------
    # 2) main loop by segments: anchor -> next_anchor (K frames ahead)
    # -------------------------------------------------------
    with torch.no_grad():
        amp_device = "cuda" if DEVICE == "cuda" else "cpu"
        with torch.amp.autocast(device_type=amp_device, enabled=(DEVICE == "cuda")):

            while processed < frames_to_propagate:
                seg_k = min(K, frames_to_propagate - processed)
                if seg_k <= 0:
                    break

                # read seg_k frames
                seg_frames = []
                for _ in range(seg_k):
                    t0 = time.perf_counter()
                    ok, fr = cap.read()
                    t_read += (time.perf_counter() - t0)
                    if (not ok) or fr is None:
                        break
                    seg_frames.append(fr)
                if len(seg_frames) == 0:
                    break

                next_anchor_frame = seg_frames[-1]

                # ---- run CUTIE on next anchor (ON ROI), independent anchor (no reinject) ----
                t0 = time.perf_counter()
                next_roi = roi
                next_anchor_roi = _crop_roi(next_anchor_frame, next_roi)
                t_roi += (time.perf_counter() - t0)

                t0 = time.perf_counter()
                fr_t = image_to_torch(next_anchor_roi, device=DEVICE)
                pred_anchor = processor.step(fr_t)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                t_model += (time.perf_counter() - t0)
                model_calls += 1

                t0 = time.perf_counter()
                next_anchor_mask_roi = torch_prob_to_numpy_mask(pred_anchor).astype(np.uint8)
                next_anchor_mask = _place_roi_mask_to_full(next_anchor_mask_roi, next_roi)
                t_misc += (time.perf_counter() - t0)

                # update ROI based on next anchor mask (follow the object)
                t0 = time.perf_counter()
                cx, cy = _center_from_mask(next_anchor_mask, cx, cy)
                roi = _roi_from_center(cx, cy)
                t_roi += (time.perf_counter() - t0)

                mid_count = len(seg_frames) - 1

                # ---- no intermediate => just append anchor ----
                if mid_count <= 0:
                    t0 = time.perf_counter()
                    overlay_frames_bgr.append(
                        overlay_fast_bgr(next_anchor_frame, next_anchor_mask, alpha=0.45, draw_outline=True)
                    )
                    t_overlay += (time.perf_counter() - t0)
                    processed += 1
                    anchor_frame = next_anchor_frame
                    anchor_mask = next_anchor_mask
                    continue

                # -------------------------------------------------------
                # forward warps: anchor -> mid frames
                # -------------------------------------------------------
                fwd_masks = [None] * mid_count
                prev_f = anchor_frame
                prev_m = anchor_mask
                for i in range(mid_count):
                    cur_f = seg_frames[i]
                    cur_m = warp_mask_lk(prev_f, cur_f, prev_m)
                    fwd_masks[i] = cur_m
                    prev_f = cur_f
                    prev_m = cur_m

                # -------------------------------------------------------
                # backward warps: next_anchor -> mid frames (reverse)
                # -------------------------------------------------------
                bwd_masks = [None] * mid_count
                prev_f = next_anchor_frame
                prev_m = next_anchor_mask
                for i in range(mid_count - 1, -1, -1):
                    cur_f = seg_frames[i]
                    cur_m = warp_mask_lk(prev_f, cur_f, prev_m)
                    bwd_masks[i] = cur_m
                    prev_f = cur_f
                    prev_m = cur_m

                # -------------------------------------------------------
                # blend masks + overlay mid frames
                # -------------------------------------------------------
                for i in range(mid_count):
                    t0 = time.perf_counter()
                    wgt = float(i + 1) / float(len(seg_frames))  # (0,1)
                    fm = fwd_masks[i].astype(np.float32)
                    bm = bwd_masks[i].astype(np.float32)
                    blended = ((1.0 - wgt) * fm + wgt * bm) >= 0.5
                    pred_index = blended.astype(np.uint8)
                    t_misc += (time.perf_counter() - t0)

                    t0 = time.perf_counter()
                    overlay_frames_bgr.append(
                        overlay_fast_bgr(seg_frames[i], pred_index, alpha=0.45, draw_outline=True)
                    )
                    t_overlay += (time.perf_counter() - t0)

                    processed += 1
                    if processed >= frames_to_propagate:
                        break

                if processed >= frames_to_propagate:
                    break

                # append next anchor frame
                t0 = time.perf_counter()
                overlay_frames_bgr.append(
                    overlay_fast_bgr(next_anchor_frame, next_anchor_mask, alpha=0.45, draw_outline=True)
                )
                t_overlay += (time.perf_counter() - t0)
                processed += 1

                # slide anchors
                anchor_frame = next_anchor_frame
                anchor_mask = next_anchor_mask

    cap.release()

    if len(overlay_frames_bgr) == 0:
        raise gr.Error("No frames processed. Check video path / start_frame.")

    # ---- save video time ----
    t0 = time.perf_counter()
    overlay_mp4 = os.path.join(OUT_DIR, f"overlay_{int(time.time()*1000)}.mp4")
    _save_overlay_video(overlay_frames_bgr, fps, overlay_mp4)
    t_save = time.perf_counter() - t0

    t_total = time.perf_counter() - t_total0

    processed = len(overlay_frames_bgr)
    video_seconds = processed / max(1e-6, fps)
    avg_gen_fps = processed / max(1e-6, t_total)
    slow_x = (t_total / max(1e-6, video_seconds))

    per = lambda x: (x / max(1, processed))
    status = (
        f"Done | processed={processed} frames | video_len={video_seconds:.2f}s @ {fps:.2f}fps\n"
        f"TOTAL: {t_total:.3f}s | avg_gen_fps={avg_gen_fps:.2f} | slow_x={slow_x:.2f}x (vs realtime)\n"
        f"Breakdown (total / per-frame):\n"
        f"  read:    {t_read:.3f}s  / {per(t_read)*1000:.2f} ms\n"
        f"  ROI:     {t_roi:.3f}s   / {per(t_roi)*1000:.2f} ms   (roi_side={roi_side})\n"
        f"  CUTIE:   {t_model:.3f}s / {per(t_model)*1000:.2f} ms   (model_calls={model_calls})\n"
        f"  LK:      {t_lk:.3f}s    / {per(t_lk)*1000:.2f} ms      (lk_pairs={lk_pairs}, lk_bad={lk_bad})\n"
        f"  overlay: {t_overlay:.3f}s / {per(t_overlay)*1000:.2f} ms\n"
        f"  misc:    {t_misc:.3f}s / {per(t_misc)*1000:.2f} ms\n"
        f"  save_mp4:{t_save:.3f}s  / {per(t_save)*1000:.2f} ms\n"
        f"Params: max_internal_size={max_internal_size} | K(lk_every)={K} | lk_corners={lk_corners} | lk_inlier={lk_inlier}\n"
        f"Video: {os.path.basename(vp)} size={w}x{h} start={start_frame_idx}"
    )

    return overlay_mp4, status






def run_track_safe(video_path_str, start_frame_idx, editor_value, frames_to_propagate, max_internal_size,
                   lk_every, lk_corners, lk_inlier):
    try:
        return run_track(video_path_str, start_frame_idx, editor_value, frames_to_propagate, max_internal_size,
                         lk_every, lk_corners, lk_inlier)

    except Exception as e:
        traceback.print_exc()
        raise gr.Error(str(e))



with gr.Blocks() as demo:
    gr.Markdown("## CUTIE (Preview video in Gradio + draw mask on a selected frame)")

    #video_path = gr.Textbox(label="Video path (in /content)", value=DEFAULT_VIDEO)
    video_upload = gr.Video(label="Upload / Preview Video", autoplay=True, loop=True, height=480)
    with gr.Row():
        load_btn = gr.Button("Load video")
        info = gr.Textbox(label="Info", interactive=False)

    with gr.Row():
        orig_video = gr.Video(label="Original Video (preview here)")
        overlay_video = gr.Video(label="Overlay Video (result preview)")

    gr.Markdown("### 1) Use the video player to preview (pause/seek).  2) Choose a frame index below to annotate (Gradio can't read the paused timestamp).")

    with gr.Row():
        frame_idx = gr.Slider(0, 0, value=0, step=1, label="Frame index to annotate (acts like pause point)")
        show_btn = gr.Button("Load this frame for annotation")

    with gr.Row():
        frame_view = gr.Image(label="Selected Frame", type="pil", interactive=False)
        mask_editor = gr.ImageEditor(label="Paint directly ON the frame (your strokes define the mask)", type="pil")

    #frames_to_prop = gr.Slider(1, 1000, value=200, step=1, label="frames_to_propagate")


    #frames_to_prop = gr.Slider(1, 1, value=1, step=1, label="frames_to_propagate (auto limited)")

    frames_to_prop = gr.Slider(1, 1, value=200, step=1, label="frames_to_propagate (auto max = remaining frames)")

    max_internal_size = gr.Slider(
    256, 1024, value=720, step=32,
    label="max_internal_size (max internal side; smaller=faster, lower quality)"
)
    lk_every = gr.Slider(1, 30, value=6, step=1, label="LK keyframe interval K (run CUTIE every K frames; 1 = pure CUTIE)")
    lk_corners = gr.Slider(50, 2000, value=400, step=50, label="LK maxCorners (more=stable, slower CPU)")
    lk_inlier = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="LK inlier_ratio threshold (lower=more warp, higher=more fallback to CUTIE)")


    run_btn = gr.Button("Run CUTIE from this frame")
    status = gr.Textbox(label="Status", interactive=False)


    load_btn.click(
    load_video,
    inputs=[video_upload],
    outputs=[orig_video, frame_view, mask_editor, frame_idx, frames_to_prop, max_internal_size, info],
    queue=False
)

    show_btn.click(
    show_frame,
    inputs=[video_upload, frame_idx],
    outputs=[frame_view, mask_editor, frames_to_prop],
    queue=False
)
    run_btn.click(
    run_track_safe,
    inputs=[video_upload, frame_idx, mask_editor, frames_to_prop, max_internal_size, lk_every, lk_corners, lk_inlier],
    outputs=[overlay_video, status],
    queue=True
)





demo.launch(allowed_paths=[OUT_DIR], inbrowser=True,debug=True)