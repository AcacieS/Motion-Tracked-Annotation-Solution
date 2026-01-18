import os, cv2, tempfile
import numpy as np
from PIL import Image
import torch
import gradio as gr
from omegaconf import open_dict
# import nest_asyncio
# nest_asyncio.apply()

from cutie.inference.inference_core import InferenceCore
from gui.interactive_utils import image_to_torch, torch_prob_to_numpy_mask, index_numpy_to_one_hot_torch, overlay_davis

# DEFAULT_VIDEO = "echo[1].mp4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

overlay_state = gr.State([])
fps_state = gr.State(0)

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

def paint_mask_red(frame_bgr, mask_index):
    """
    frame_bgr : original frame in BGR (OpenCV)
    mask_index: numpy array HxW with 1 where object is
    """
    out = frame_bgr.copy()
    red = np.zeros_like(out)
    red[:] = (0, 0, 255)  # BGR pure red

    alpha = 0.5  # transparency
    mask = mask_index.astype(bool)

    out[mask] = (alpha * red[mask] + (1 - alpha) * out[mask]).astype(np.uint8)
    return out

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

def _mask_from_editor(editor_value):
    if editor_value is None:
        raise gr.Error("Mask editor is empty. Please paint on the frame.")

    bg = editor_value.get("background", None)
    comp = editor_value.get("composite", None)

    if bg is None:
        raise gr.Error("ImageEditor returned no background.")

    # If composite missing, fall back to background
    if comp is None:
        comp = bg

    # Convert numpy → PIL if needed
    if isinstance(bg, np.ndarray):
        bg = Image.fromarray(bg)
    if isinstance(comp, np.ndarray):
        comp = Image.fromarray(comp)

    bg = bg.convert("RGB")
    comp = comp.convert("RGB")

    bg_arr = np.array(bg).astype(np.int16)
    cp_arr = np.array(comp).astype(np.int16)

    if bg_arr.shape != cp_arr.shape:
        raise gr.Error("Editor output size mismatch. Try reloading the frame.")

    diff = np.abs(cp_arr - bg_arr).sum(axis=-1)
    mask = (diff > 25).astype(np.uint8)
    return mask

def _save_overlay_video(frames_bgr, fps, out_mp4):
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))
    for f in frames_bgr:
        vw.write(f)
    vw.release()

def load_video(video_path_str):
    # vp = _resolve_video_path(video_path_str)
    vp = video_path_str
    if vp is None:
        raise gr.Error("Please upload a video.")
    fps, n, w, h = _get_video_info(vp)
    _, frame0_pil = _read_frame(vp, 0)
    editor_init = _editor_value_from_frame(frame0_pil)
    info = f"Loaded: {vp} | frames={n} | fps={fps:.2f} | size={w}x{h}"
    # 更新 slider 的最大值
    slider_update = gr.Slider(minimum=0, maximum=max(0, n-1), value=0, step=1)
    return vp, frame0_pil, editor_init, slider_update, info

def show_frame(video_path_str, frame_idx):
    # vp = _resolve_video_path(video_path_str)
    vp = video_path_str
    _, frame_pil = _read_frame(vp, int(frame_idx))
    # 切帧时：编辑器背景同步为这帧（你就在这帧上画）
    return frame_pil, _editor_value_from_frame(frame_pil)

# def run_track(video_path_str, start_frame_idx, editor_value, frames_to_propagate):
def run_track(video_path_str, start_frame_idx, editor_value, frames_to_propagate, max_internal_size):
    max_internal_size = int(max_internal_size)
    # vp = _resolve_video_path(video_path_str)
    vp = video_path_str
    fps, n, w, h = _get_video_info(vp)

    start_frame_idx = int(start_frame_idx)
    frames_to_propagate = int(frames_to_propagate)

    # 从 editor 得到 index mask：0/1
    mask_index = _mask_from_editor(editor_value)
    if mask_index.sum() < 10:
        raise gr.Error("Mask too small / empty. Please paint a larger region on the frame.")

    num_objects = 1
    processor = InferenceCore(cutie, cfg=cfg)

    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        raise gr.Error(f"Cannot open video: {vp}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

    overlay_frames_bgr = []
    current = 0
    torch.cuda.empty_cache()

    with torch.no_grad():
        #torch.set_grad_enabled(False)
        while cap.isOpened():
            ok, frame = cap.read()
            if (not ok) or frame is None or current >= frames_to_propagate:
                break

            frame_torch = image_to_torch(frame, device=DEVICE)

            if current == 0:
                mask_torch = index_numpy_to_one_hot_torch(mask_index, num_objects + 1).to(DEVICE)
                pred = processor.step(frame_torch, mask_torch[1:], idx_mask=False)
            else:
                pred = processor.step(frame_torch)

            pred_index = torch_prob_to_numpy_mask(pred)

            # frame is BGR from OpenCV already
            vis_bgr = paint_mask_red(frame, pred_index)
                        #getting each frame together
            overlay_frames_bgr.append(vis_bgr)
            current += 1
            
            #showing each frame
            vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
            yield vis_rgb, None, f"Processing frame {current}..."

    cap.release()

    if len(overlay_frames_bgr) == 0:
        raise gr.Error("No frames processed.")

    status = f"Done streaming {current} frames. Encoding video..."
    return overlay_frames_bgr, fps, status

def finalize_video(frames_bgr, fps, status):
    out_dir = tempfile.mkdtemp(prefix="cutie_ui_")
    overlay_mp4 = os.path.join(out_dir, "overlay.mp4")
    _save_overlay_video(frames_bgr, fps, overlay_mp4)
    return None, overlay_mp4, status.replace("Encoding", "Done")

with gr.Blocks() as demo:
    gr.Markdown("## CUTIE (Preview video in Gradio + draw mask on a selected frame)")

    video_upload = gr.Video(label="Upload a video")
    with gr.Row():
        load_btn = gr.Button("Load video")
        info = gr.Textbox(label="Info", interactive=False)

    with gr.Row():
        orig_video = gr.Video(label="Original Video (preview here)")
        overlay_video = gr.Video(label="Final Overlay Video (result preview)")
        live_preview = gr.Image(label="Live CUTIE Output")

    gr.Markdown("### 1) Use the video player to preview (pause/seek).  2) Choose a frame index below to annotate (Gradio can't read the paused timestamp).")

    with gr.Row():
        frame_idx = gr.Slider(0, 0, value=0, step=1, label="Frame index to annotate (acts like pause point)")
        show_btn = gr.Button("Load this frame for annotation")

    with gr.Row():
        frame_view = gr.Image(label="Selected Frame", type="pil", interactive=False)
        mask_editor = gr.ImageEditor(label="Paint directly ON the frame (your strokes define the mask)")

    frames_to_prop = gr.Slider(1, 1000, value=200, step=1, label="frames_to_propagate")
    max_internal_size = gr.Slider(
        256, 1024, value=720, step=32,
        label="max_internal_size (lower=faster, higher=sharper)"
    )
    run_btn = gr.Button("Run CUTIE from this frame")
    status = gr.Textbox(label="Status", interactive=False)

    
    load_btn.click(load_video, inputs=[video_upload],
                   outputs=[orig_video, frame_view, mask_editor, frame_idx, info],
                   queue=False)

    show_btn.click(show_frame, inputs=[video_upload, frame_idx],
                   outputs=[frame_view, mask_editor],
                   queue=False)
    
    run_event = run_btn.click(
            run_track,
            inputs=[video_upload, frame_idx, mask_editor, frames_to_prop, max_internal_size],
            outputs=[live_preview, overlay_video, status],
            queue=True
        )
    run_event.then(
    finalize_video,
    inputs=[live_preview, overlay_video, status],
    outputs=[live_preview, overlay_video, status]
)

    # run_btn.click(run_track, inputs=[video_upload, frame_idx, mask_editor, frames_to_prop],
    #               outputs=[live_preview, overlay_video, status],
    #               queue=True)

#demo.launch(share=True, debug=True, prevent_thread_lock=True)
demo.queue(max_size=1, default_concurrency_limit=1).launch(share=True, debug=True)