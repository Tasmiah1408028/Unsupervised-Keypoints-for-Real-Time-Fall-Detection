# --- CUDA allocator env fix (must be before torch import) ---
import os
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

import time
import cv2
import numpy as np
from collections import deque
from pathlib import Path

from ultralytics import YOLO
import torch
# =========================
# CONFIG
# =========================
MODEL_WEIGHTS = "yolov8m-seg.pt"  # instance segmentation head
INPUT_ROOT   = '../URFall'   # folder of subfoldercd ..s, each subfolder = 1 video
OUTPUT_ROOT  = '../URFall_seg'

DEVICE = 0 if torch.cuda.is_available() else "cpu"
IMG_SIZE = 800 #upgraded image sizing to 800 form 480
FRAME_STRIDE = 1
LIMIT_VIDEOS = None
SKIP_IF_EXISTS = True

# Tracker
IOU_MATCH_THRESH = 0.20 # tracks change from box of current frame to new frame set for fast motion
EMA_ALPHA        = 0.80 # smooths the movement of the box
AREA_GROWTH_CAP  = 1.25 # limits box expansion per frame
AREA_SHRINK_CAP  = 0.65 # limits box shrinkage per frame
MAX_AGE          = 30  #1 secdon of frame memory 30 FPS

# Mask stabilizers
MIN_PERSON_CONF     = 0.18       # a bit lower so first frames don't miss. what qualifies as detection
MIN_MASK_AREA_FRAC  = 0.003     # 0.5% of track ROI
MAX_MASK_AREA_FRAC  = 0.92       # allow big subjects
TEMPORAL_WINDOW     = 5
MEDIAN_ADOPT_IOU    = 0.30

OVERLAY_ALPHA = 0.35
BOX_COLOR     = (0,255,0)
BOX_THICKNESS = 2

MASK_PAD_PX = 5   
FEATHER_PX  = 2

# =============== UTILS ===============
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def list_sequences(root):
    out={}
    for cr, _, files in os.walk(root):
        imgs=[f for f in files if f.lower().endswith((".jpg",".jpeg",".png"))]
        if not imgs: continue
        imgs.sort()
        rel=os.path.relpath(cr, root)
        out[rel]=[os.path.join(cr,f) for f in imgs]
    return out

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1,ix2,iy2=max(ax1,bx1),max(ay1,by1),min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1)
    inter=iw*ih
    ua=max(1,(ax2-ax1)*(ay2-ay1)) + max(1,(bx2-bx1)*(by2-by1)) - inter
    return inter/ua

def clamp_growth(prev_b, cur_b):
    pa=max(1,(prev_b[2]-prev_b[0])*(prev_b[3]-prev_b[1]))
    ca=max(1,(cur_b[2]-cur_b[0])*(cur_b[3]-cur_b[1]))
    if ca > pa*AREA_GROWTH_CAP:
        scale=(pa*AREA_GROWTH_CAP/ca)**0.5
    elif ca < pa*AREA_SHRINK_CAP:
        scale=(pa*AREA_SHRINK_CAP/ca)**0.5
    else:
        scale=1.0
    if scale!=1.0:
        cx=(cur_b[0]+cur_b[2])*0.5; cy=(cur_b[1]+cur_b[3])*0.5
        w=(cur_b[2]-cur_b[0])*scale; h=(cur_b[3]-cur_b[1])*scale
        cur_b=[int(cx-w*0.5),int(cy-h*0.5),int(cx+w*0.5),int(cy+h*0.5)]
    return cur_b

def ema_box(prev_b, det_b, a=EMA_ALPHA):
    pcx=(prev_b[0]+prev_b[2])*0.5; pcy=(prev_b[1]+prev_b[3])*0.5
    pw=prev_b[2]-prev_b[0]; ph=prev_b[3]-prev_b[1]
    dcx=(det_b[0]+det_b[2])*0.5; dcy=(det_b[1]+det_b[3])*0.5
    dw=det_b[2]-det_b[0]; dh=det_b[3]-det_b[1]
    cx=a*dcx+(1-a)*pcx; cy=a*dcy+(1-a)*pcy
    w=a*dw+(1-a)*pw; h=a*dh+(1-a)*ph
    return [int(cx-w*0.5),int(cy-h*0.5),int(cx+w*0.5),int(cy+h*0.5)]

def ensure_mask_uint8_hw(mask, H, W):
    if mask is None:
        return np.zeros((H,W), np.uint8)
    m = np.asarray(mask)
    if m.ndim==3:
        if m.shape[0]==1: m=m[0]
        if m.shape[-1]==1: m=m[...,0]
    if m.shape[:2] != (H,W):
        m = cv2.resize(m.astype(np.float32), (W,H), interpolation=cv2.INTER_NEAREST)
    if m.dtype != np.float32 and m.dtype != np.float64:
        m = m.astype(np.float32)
    return (m>=0.5).astype(np.uint8)

def overlay_mask_on_frame(bgr, mask01, alpha=OVERLAY_ALPHA):
    H,W = bgr.shape[:2]
    mask01 = ensure_mask_uint8_hw(mask01, H, W)
    overlay = bgr.copy()
    color = np.array([0,0,255], np.uint8)
    idx = mask01>0
    overlay[idx] = (overlay[idx]*(1-alpha) + color*alpha).astype(np.uint8)
    return overlay

def draw_box(bgr, box, color=BOX_COLOR, thickness=BOX_THICKNESS):
    if box is None: return bgr
    x1,y1,x2,y2 = map(int, box)
    x1 = max(0,min(x1,bgr.shape[1]-1))
    y1 = max(0,min(y1,bgr.shape[0]-1))
    x2 = max(0,min(x2,bgr.shape[1]-1))
    y2 = max(0,min(y2,bgr.shape[0]-1))
    out = bgr.copy()
    cv2.rectangle(out,(x1,y1),(x2,y2),color,thickness)
    return out

def make_debug_panel(orig, overlay, mask01):
    H,W = orig.shape[:2]
    mask01 = ensure_mask_uint8_hw(mask01, H, W)
    m3 = cv2.cvtColor(mask01*255, cv2.COLOR_GRAY2BGR)
    return np.concatenate([orig, overlay, m3], axis=1)

def box_from_mask(m):
    ys, xs = np.where(m>0)
    if xs.size==0: return None
    return [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]

# =============== TRACKER ===============
class SimpleTrack:
    def __init__(self):
        self.box=None
        self.age=0
    def update(self, det_boxes):
        if self.box is None:
            if not det_boxes: return False
            self.box = max(det_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            self.age=0
            return True
        # match
        best=None; best_i=IOU_MATCH_THRESH
        for b in det_boxes:
            iv=iou(self.box,b)
            if iv>best_i: best_i=iv; best=b
        if best is not None:
            sm=ema_box(self.box,best,EMA_ALPHA)
            sm=clamp_growth(self.box,sm)
            self.box=sm; self.age=0
            return True
        # predict/hold
        self.age+=1
        if self.age>MAX_AGE: return False
        return True

# =============== MAIN ===============
def main():
    ensure_dir(OUTPUT_ROOT)
    print(f"Loading YOLO: {MODEL_WEIGHTS}")
    model = YOLO(MODEL_WEIGHTS)
    try:
        model.to("cuda" if DEVICE==0 else DEVICE)
        device_arg = 0 if DEVICE==0 else ("cpu" if DEVICE=="cpu" else DEVICE)
    except Exception as e:
        print(f"⚠️ model.to({DEVICE}) failed: {e}; falling back to CPU")
        model.to("cpu"); device_arg = "cpu"

    vids = list_sequences(INPUT_ROOT)
    if not vids:
        print("No input frames found."); return
    rel_dirs = list(vids.keys())
    if LIMIT_VIDEOS is not None:
        rel_dirs = rel_dirs[:LIMIT_VIDEOS]

    for vi, rel in enumerate(rel_dirs,1):
        frames_all = vids[rel]
        frames = frames_all[::FRAME_STRIDE]
        out_dir = os.path.join(OUTPUT_ROOT, rel); ensure_dir(out_dir)
        print(f"▶ [{vi}/{len(rel_dirs)}] {rel} — {len(frames)} frames")

        track = SimpleTrack()
        prev_mask = None
        mask_window = deque(maxlen=TEMPORAL_WINDOW)
        t0=time.time()

        for idx, fp in enumerate(frames,1):
            base = Path(fp).stem
            mask_path   = os.path.join(out_dir, f"{base}_mask.png")
            overlay_path= os.path.join(out_dir, f"{base}_overlay.png")
            panel_path  = os.path.join(out_dir, f"{base}_panel.jpg")
            if SKIP_IF_EXISTS and os.path.isdir(out_dir):
                done_subjects = sum(
                    1 for f in os.listdir(out_dir)
                    if f.endswith("_subject.jpg")
                )
                if done_subjects >= len(frames):
                    print(f"⏭  Skipping '{rel}': found {done_subjects}/{len(frames)} subject frames.")
                    continue

            img = cv2.imread(fp)
            if img is None:
                print(f"❌ unreadable {fp}"); continue
            H,W = img.shape[:2]

            # --- Predict
            res = model.predict(img, device=device_arg, imgsz=IMG_SIZE, verbose=False)[0]

            det_boxes=[]; det_masks=[]
            if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes)>0:
                cls  = res.boxes.cls.cpu().numpy().astype(int)
                conf = res.boxes.conf.cpu().numpy()
                xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
                # masks: Ultralytics returns upscaled (N,H,W) for seg models
                up = None
                if hasattr(res, "masks") and res.masks is not None:
                    # robust extraction across versions
                    try:
                        up = res.masks.data  # torch.Tensor (N,H,W)
                    except Exception:
                        up = None
                if up is not None:
                    up = up.cpu().numpy()
                keep = [i for i,(c,cf) in enumerate(zip(cls,conf)) if c==0 and cf>=MIN_PERSON_CONF]
                for i in keep:
                    det_boxes.append(xyxy[i].tolist())
                    if up is not None:
                        det_masks.append((up[i] > 0.5).astype(np.uint8))

            # --- Fallback: if we got masks but no boxes (rare API quirk)
            if not det_boxes and det_masks:
                # init from largest mask bbox
                bbs = [box_from_mask(m) for m in det_masks]
                bbs = [b for b in bbs if b is not None]
                if bbs:
                    det_boxes = bbs

            # --- Update tracker
            ok = track.update(det_boxes)
            if not ok:
                track = SimpleTrack()
                cur_mask = prev_mask if prev_mask is not None else np.zeros((H,W),np.uint8)
                chosen_idx = -1
            else:
                # choose mask overlapping tracked box best (by mask bbox IoU)
                chosen = None; chosen_idx=-1; best_i=0.0
                if det_masks and track.box is not None:
                    for i,m in enumerate(det_masks):
                        mb = box_from_mask(m)
                        if mb is None: continue
                        iv = iou(track.box, mb)  # use mask bbox overlap
                        if iv>best_i: best_i=iv; chosen=m; chosen_idx=i
                if chosen is None and det_masks:
                    # fallback: take largest mask
                    areas = [int(m.sum()) for m in det_masks]
                    chosen_idx = int(np.argmax(areas))
                    chosen = det_masks[chosen_idx]
                if chosen is None and det_masks:
                    # last fallback: union all masks
                    chosen = (np.sum(np.stack(det_masks,0), axis=0) > 0).astype(np.uint8)

                if chosen is None:
                    cur_mask = prev_mask if prev_mask is not None else np.zeros((H,W),np.uint8)
                else:
                    # sanity in track ROI
                    x1,y1,x2,y2 = track.box if track.box is not None else [0,0,W,H]
                    x1=max(0,min(x1,W-1)); x2=max(1,min(x2,W))
                    y1=max(0,min(y1,H-1)); y2=max(1,min(y2,H))
                    roi = chosen[y1:y2, x1:x2]
                    roi_area = max(1,(x2-x1)*(y2-y1))
                    frac = float(roi.sum())/float(roi_area)
                    if frac < MIN_MASK_AREA_FRAC or frac > MAX_MASK_AREA_FRAC:
                        # implausible -> use union of all to avoid all-black
                        if det_masks:
                            union = (np.sum(np.stack(det_masks,0), axis=0) > 0).astype(np.uint8)
                            cur_mask = union
                            chosen_idx = -2
                        else:
                            cur_mask = prev_mask if prev_mask is not None else np.zeros((H,W),np.uint8)
                    else:
                        cur_mask = chosen

            # --- Temporal median (gentle, no overgrowth)
            cur_mask = ensure_mask_uint8_hw(cur_mask, H, W)

            if MASK_PAD_PX > 0:
                k = max(1, int(MASK_PAD_PX))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
                cur_mask = cv2.dilate(cur_mask, kernel, iterations=1)
            
            if FEATHER_PX > 0:
                # optional: soften overlay edge only (mask stays binary for cut-outs)
                pass

            
            mask_window.append(cur_mask.copy())
            if len(mask_window) >= 3:
                stack = np.stack(mask_window,0).astype(np.uint8)
                med = (np.median(stack,0) >= 0.5).astype(np.uint8)
                inter = np.logical_and(med>0, cur_mask>0).sum()
                uni   = np.logical_or (med>0, cur_mask>0).sum()+1e-6
                if inter/uni >= MEDIAN_ADOPT_IOU and med.sum() <= int(1.3*cur_mask.sum()+1):
                    cur_mask = med

            # --- Visualization (never black/white canvas)
            overlay = overlay_mask_on_frame(img, cur_mask, alpha=OVERLAY_ALPHA)
            overlay = draw_box(overlay, track.box)
            panel   = make_debug_panel(img, overlay, cur_mask)

            # --- Save
            ensure_dir(out_dir)
            subject_bgr = img.copy()
            subject_bgr[cur_mask == 0] = 0
            cv2.imwrite(os.path.join(out_dir, f"{base}_subject.jpg"), subject_bgr)

            # --- Debug line (helps verify no-all-black)
            nz = int(cur_mask.sum())
            n_det = len(det_boxes)
            n_msk = len(det_masks)
            print(f"{rel} f{idx:04d}: det={n_det} masks={n_msk} chosen={chosen_idx} mask_nz={nz}")

            prev_mask = cur_mask

            if idx % 25 == 0 or idx==len(frames):
                fps = idx/(time.time()-t0+1e-6)
                print(f"{rel}: {idx}/{len(frames)} ~ {fps:.2f} fps")

    print("\n===== DONE =====")
    print("Output:", os.path.abspath(OUTPUT_ROOT))

if __name__ == "__main__":
    main()