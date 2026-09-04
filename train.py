from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, glob, random
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold

# ---------------- Determinism knobs (best-effort) ----------------
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from absl import app
from absl import flags

import tensorflow.compat.v1 as tf

import datasets_for_classifier
import dynamics
import hyperparameters
import losses
import vision

FLAGS = flags.FLAGS


# ---------------- Helpers: seed, file listing, labels, k-fold ----------------
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    # TF2 determinism API (works in some installs; safe to ignore if not present)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def list_npz_files(dir_path):
    return sorted(glob.glob(os.path.join(dir_path, "*.npz")))


def label_from_name(path):
    b = os.path.basename(path).lower()
    if b.startswith("adl-"):
        return 0
    if b.startswith("fall-"):
        return 1
    raise ValueError("Unknown label prefix for file: {}".format(b))

def make_label_map(files):
    # map basename -> int32 label
    return {os.path.basename(f): np.int32(label_from_name(f)) for f in files}

def identity_from_name(f):
    basename = os.path.basename(f)
    match = re.search(r'fall-(\d+)-', basename, re.IGNORECASE)
    if match:
        num = int(match.group(1))
        if num <= 6:
            return "ID1"
        elif 7 <= num <= 12:
            return "ID2"
        elif 13 <= num <= 18:
            return "ID3"
        else:
            return None  # test identities, should not appear in train
    return None  # ADL or no match

def stratified_kfold_identity(files, k=5, seed=0):
    """
    files: list of all file paths
    fall_identities: dict mapping fall video filename -> identity string
                     e.g. {"fall_001.mp4": "ID1", "fall_002.mp4": "ID1", ...}
                     ADL files not in this dict
    """
    files = np.array(files)
    labels = np.array([label_from_name(f) for f in files], dtype=np.int32)
    
    # Build strata
    strata = []
    for f, lab in zip(files, labels):
        if lab == 1:  # fall video
            identity = identity_from_name(f)
            strata.append(f"fall_{identity}")
        else:
            strata.append("adl")
    strata = np.array(strata)
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for train_idx, val_idx in skf.split(files, strata):
        yield files[train_idx].tolist(), files[val_idx].tolist()


def make_bag_dataset(cfg, files, repeat_dataset, shuffle_files, shuffle_bags):
    """Build bag_by_video dataset from an explicit file list."""
    label_map = make_label_map(files)

    ds, shapes = datasets_for_classifier.get_sequence_dataset(
        data_dir=None,  # unused when filenames_override is provided
        batch_size=cfg.batch_size,  # ignored for bag_by_video=True (no batching)
        num_timesteps=24, #cfg.observed_steps + cfg.predicted_steps,
        random_offset=False,          # keep stable
        repeat_dataset=repeat_dataset,
        seed=cfg.seed,
        num_adl=None, num_fall=None,  # IMPORTANT for k-fold (do NOT use fixed counts)
        bag_by_video=True,
        max_chunks_per_video=getattr(cfg, "max_chunks_per_video", 25),
        # NEW ARGS (require patch in datasets_for_classifier.py)
        filenames_override=files,
        label_map_override=label_map,
        shuffle_files=shuffle_files,
        shuffle_bags=shuffle_bags,
    )
    return ds, shapes


def softmax_np(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def eval_video_level(model, ds, num_videos):
    """Evaluate per-video accuracy + confusion matrix on a bag_by_video dataset."""
    y_true = []
    y_pred = []

    # rows=true [ADL,Fall], cols=pred [ADL,Fall]
    cm = np.zeros((2, 2), dtype=np.int32)

    n = 0
    for d in ds:
        images = d["image"].numpy()     # (num_chunks, L, H, W, C)
        labels = d["label"].numpy()     # (num_chunks,)
        fn = d["filename"].numpy()      # (num_chunks,)

        true = int(labels[0])

        outs = model.predict_on_batch([images, labels])
        logits_video = outs[-1]         # (1,2)
        prob = softmax_np(logits_video)[0]
        pred = int(np.argmax(prob))

        y_true.append(true)
        y_pred.append(pred)
        cm[true, pred] += 1

        n += 1
        if n >= num_videos:
            break

    acc = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0
    return acc, cm

def _cummax_axis1(x):
    """
    x: [N, T] int32/float32
    returns: cumulative max along T (axis=1), same shape
    """
    xt = tf.transpose(x, [1, 0])  # [T, N]
    out_t = tf.scan(
        lambda a, b: tf.maximum(a, b),
        xt,
        initializer=xt[0],
        parallel_iterations=1
    )
    return tf.transpose(out_t, [1, 0])  # [N, T]

def _cummin_axis1(x):
    """
    x: [N, T] int32/float32
    returns: cumulative min along T (axis=1), same shape
    """
    xt = tf.transpose(x, [1, 0])  # [T, N]
    out_t = tf.scan(
        lambda a, b: tf.minimum(a, b),
        xt,
        initializer=xt[0],
        parallel_iterations=1
    )
    return tf.transpose(out_t, [1, 0])  # [N, T]

def _interp_1d_with_mask(values, valid):
    """
    values: [N, T] float32
    valid : [N, T] bool  (True where joint/kp is present)
    Returns: [N, T] float32 where invalid points are linearly interpolated.
    """
    values = tf.convert_to_tensor(values)
    valid  = tf.convert_to_tensor(valid)

    N = tf.shape(values)[0]
    T = tf.shape(values)[1]

    t_idx = tf.range(T, dtype=tf.int32)[None, :]         # [1, T]
    t_idx_tile = tf.tile(t_idx, [N, 1])                  # [N, T]

    # prev index: if valid -> t else -1, then cummax
    idx_prev_src = tf.where(valid, t_idx_tile, tf.fill([N, T], tf.constant(-1, tf.int32)))
    idx_prev = _cummax_axis1(idx_prev_src)               # [N, T]

    # next index: if valid -> t else T, then reverse + cummin + reverse
    idx_next_src = tf.where(valid, t_idx_tile, tf.fill([N, T], T))
    idx_next = tf.reverse(
        _cummin_axis1(tf.reverse(idx_next_src, axis=[1])),
        axis=[1]
    )                                                    # [N, T]

    # Gather prev/next values using gather_nd (TF1-safe)
    row = tf.tile(tf.expand_dims(tf.range(N, dtype=tf.int32), 1), [1, T])  # [N, T]

    prev_clip = tf.maximum(idx_prev, 0)
    next_clip = tf.minimum(idx_next, T - 1)

    prev_val = tf.gather_nd(values, tf.stack([row, prev_clip], axis=-1))  # [N, T]
    next_val = tf.gather_nd(values, tf.stack([row, next_clip], axis=-1))  # [N, T]

    has_prev = idx_prev >= 0
    has_next = idx_next < T

    # linear interpolation where both neighbors exist
    t_f    = tf.cast(t_idx_tile, tf.float32)
    prev_f = tf.cast(idx_prev, tf.float32)
    next_f = tf.cast(idx_next, tf.float32)

    denom = tf.maximum(next_f - prev_f, 1.0)
    w = (t_f - prev_f) / denom
    interp = prev_val + w * (next_val - prev_val)

    # boundary fallbacks
    filled = tf.where(has_prev & has_next, interp,
             tf.where(has_prev, prev_val,
             tf.where(has_next, next_val, values)))

    out = tf.where(valid, values, filled)
    return out

def linear_interpolate_keypoints_xy(kp, mu_thresh=0.05, keep_mu=True):
    """
    kp: [B, T, K, 3] where kp[...,0]=x, kp[...,1]=y, kp[...,2]=mu
    Interpolates x,y where mu < mu_thresh.
    """
    kp = tf.convert_to_tensor(kp)
    x  = kp[..., 0]
    y  = kp[..., 1]
    mu = kp[..., 2]

    valid = mu > mu_thresh  # [B, T, K] bool

    B = tf.shape(kp)[0]
    T = tf.shape(kp)[1]
    K = tf.shape(kp)[2]

    # reshape to [B*K, T]
    x2 = tf.reshape(tf.transpose(x,  [0, 2, 1]), [B * K, T])
    y2 = tf.reshape(tf.transpose(y,  [0, 2, 1]), [B * K, T])
    v2 = tf.reshape(tf.transpose(valid, [0, 2, 1]), [B * K, T])

    x2_f = _interp_1d_with_mask(x2, v2)
    y2_f = _interp_1d_with_mask(y2, v2)

    # back to [B, T, K]
    x_f = tf.transpose(tf.reshape(x2_f, [B, K, T]), [0, 2, 1])
    y_f = tf.transpose(tf.reshape(y2_f, [B, K, T]), [0, 2, 1])

    if keep_mu:
        mu_f = mu
    else:
        mu_f = tf.where(valid, mu, tf.zeros_like(mu))

    return tf.stack([x_f, y_f, mu_f], axis=-1)  # [B, T, K, 3]

def normalize_kp_video(kp_video, eps=1e-6, center_over_keypoints=True):
    """
    kp_video: [1, T, K, 3]  (x,y,mu)
    Returns same shape, with x/y centered and scaled.
    """
    x = kp_video[..., 0]
    y = kp_video[..., 1]
    mu = kp_video[..., 2]

    if center_over_keypoints:
        # mean over time AND keypoints
        mx = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        my = tf.reduce_mean(y, axis=[1, 2], keepdims=True)
        x0 = x - mx
        y0 = y - my

        # scale as RMS over time+keypoints
        sx = tf.sqrt(tf.reduce_mean(tf.square(x0), axis=[1, 2], keepdims=True) + eps)
        sy = tf.sqrt(tf.reduce_mean(tf.square(y0), axis=[1, 2], keepdims=True) + eps)
    else:
        # mean over time only (per keypoint)
        mx = tf.reduce_mean(x, axis=1, keepdims=True)
        my = tf.reduce_mean(y, axis=1, keepdims=True)
        x0 = x - mx
        y0 = y - my
        sx = tf.sqrt(tf.reduce_mean(tf.square(x0), axis=1, keepdims=True) + eps)
        sy = tf.sqrt(tf.reduce_mean(tf.square(y0), axis=1, keepdims=True) + eps)

    x_norm = x0 / sx
    y_norm = y0 / sy

    return tf.stack([x_norm, y_norm, mu], axis=-1)

class SaveLastBest(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor="val_cls_acc_video", mode="max", atol=1e-8, verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.atol = atol
        self.verbose = verbose
        self.best = -np.inf if mode == "max" else np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor, None)
        if current is None:
            return

        current = float(current)

        if self.mode == "max":
            improved = current > self.best + self.atol
            tied = np.isclose(current, self.best, atol=self.atol)
        else:
            improved = current < self.best - self.atol
            tied = np.isclose(current, self.best, atol=self.atol)

        # Save if improved OR tied with best (=> last-best behavior)
        if improved or tied:
            if improved:
                self.best = current  # update best only when strictly better
            self.model.save_weights(self.filepath)
            if self.verbose:
                print(f"\n[SaveLastBest] Saved epoch {epoch+1} "
                      f"({self.monitor}={current:.6f}, best={self.best:.6f}) -> {self.filepath}")


def _decode_name(x):
    # x can be tf.Tensor of dtype string/bytes or raw bytes
    try:
        x = x.numpy()
    except Exception:
        pass
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, str):
        return x
    return str(x)


def collect_probs_video_level(model, ds_eval, n_videos):
    """
    Collect y_true and probability of class 1 (Fall) for n_videos.

    Supports:
      (A) bag_by_video=True: dataset yields dict with keys: image,label,(filename)
          - image:   (num_chunks, L, H, W, C)
          - label:   (num_chunks,)  (same label repeated for all chunks in the bag)
          - filename:(num_chunks,)  (optional)

      (B) non-bag: dataset yields (x,y) or (x,y,fn)
          - x: (B, T, F) and model(x)->logits (B,2)
    Returns:
      y_true_list, p_fall_list, names_list
    """
    y_true, p_fall, names = [], [], []
    seen = 0

    for batch in ds_eval:
        if seen >= n_videos:
            break

        # --------------------------
        # Case A: dict batch (bag_by_video)
        # --------------------------
        if isinstance(batch, dict):
            images = batch["image"]    # (num_chunks, L, H, W, C)
            labels = batch["label"]    # (num_chunks,)
            fn     = batch.get("filename", None)

            labels_np = labels.numpy()
            true = int(labels_np[0])  # one label per video

            # Your model expects [images, labels]
            outs = model.predict_on_batch([images, labels])
            logits_video = outs[-1]          # (1,2)
            prob = tf.nn.softmax(logits_video, axis=-1).numpy()[0]
            p1 = float(prob[1])

            if fn is None:
                name = ""
            else:
                fn_np = fn.numpy()
                # take the first chunk filename as the video identifier
                name = _decode_name(fn_np[0]) if fn_np.shape else _decode_name(fn_np)

            y_true.append(true)
            p_fall.append(p1)
            names.append(name)
            seen += 1
            continue

        # --------------------------
        # Case B: tuple/list batch (non-bag)
        # --------------------------
        if isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                x, y, fn = batch
            elif len(batch) == 2:
                x, y = batch
                fn = None
            else:
                raise ValueError(f"Unexpected tuple batch length: {len(batch)}")

            x_np = x.numpy()
            if x_np.ndim == 2:  # (T,F) -> (1,T,F)
                x_np = x_np[None, ...]

            y_np = y.numpy()
            if np.ndim(y_np) == 0:
                y_np = np.array([int(y_np)], dtype=np.int32)
            else:
                y_np = y_np.astype(np.int32)

            if fn is None:
                fn_list = [""] * x_np.shape[0]
            else:
                fn_np = fn.numpy()
                if np.ndim(fn_np) == 0:
                    fn_list = [_decode_name(fn_np)]
                else:
                    fn_list = [_decode_name(f) for f in fn_np]

            logits = model(x_np, training=False).numpy()         # (B,2)
            probs = tf.nn.softmax(logits, axis=-1).numpy()       # (B,2)
            p1 = probs[:, 1]                                     # P(Fall)

            for i in range(x_np.shape[0]):
                if seen >= n_videos:
                    break
                y_true.append(int(y_np[i]))
                p_fall.append(float(p1[i]))
                names.append(fn_list[i])
                seen += 1

            continue

        raise ValueError(f"Unexpected batch type from ds_eval: {type(batch)}")

    return y_true, p_fall, names


def metrics_from_probs(y_true, p_fall, threshold=0.5):
    """
    Convert probabilities to predicted labels using threshold on class-1 prob:
      pred = 1 if p_fall >= threshold else 0
    Returns same tuple as compute_metrics: (acc, prec, rec, f1, cm)
    """
    p_fall = np.array(p_fall, dtype=np.float32)
    y_pred = (p_fall >= threshold).astype(int)
    return compute_metrics(y_true, y_pred)

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    acc = float(np.mean(y_true == y_pred))
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # class 1 = Fall
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]

    prec = float(tp / (tp + fp + 1e-9))
    rec = float(tp / (tp + fn + 1e-9))
    f1 = float(2 * prec * rec / (prec + rec + 1e-9))
    return acc, prec, rec, f1, cm


def find_best_threshold_by_f1(y_true, p_fall, num_thresholds=1001):
    """
    Finds threshold in [0,1] maximizing F1 on validation.
    Tie-break: higher recall -> then threshold closer to 0.5.
    Returns: best_thr, (acc,prec,rec,f1,cm) at best_thr
    """
    thresholds = np.linspace(0.0, 1.0, num_thresholds, dtype=np.float32)

    best_thr = 0.5
    best_acc = -1.0
    best_prec = -1.0
    best_rec = -1.0
    best_f1 = -1.0
    best_cm = None

    for thr in thresholds:
        acc, prec, rec, f1, cm = metrics_from_probs(y_true, p_fall, threshold=float(thr))

        better = False
        if f1 > best_f1 + 1e-12:
            better = True
        elif abs(f1 - best_f1) <= 1e-12:
            # tie-break 1: higher recall
            if rec > best_rec + 1e-12:
                better = True
            elif abs(rec - best_rec) <= 1e-12:
                # tie-break 2: closer to 0.5
                if abs(thr - 0.5) < abs(best_thr - 0.5):
                    better = True

        if better:
            best_thr = float(thr)
            best_acc, best_prec, best_rec, best_f1 = acc, prec, rec, f1
            best_cm = cm

    return best_thr, (best_acc, best_prec, best_rec, best_f1, best_cm)


# ---------------- Model ----------------
def build_model(cfg, data_shapes):
    input_shape_no_batch = data_shapes["image"][1:]
    input_images = tf.keras.Input(shape=input_shape_no_batch, name="image")

    # Vision model
    observed_keypoints, aux = vision.build_images_to_keypoints_net(
        cfg, input_shape_no_batch
    )(input_images)

    # mu for sparsity
    if isinstance(aux, dict) and "mu" in aux:
        mu = aux["mu"]
    else:
        mu = observed_keypoints[..., 2]

    keypoints_to_images_net = vision.build_keypoints_to_images_net(
        cfg, input_shape_no_batch
    )
    reconstructed_images = keypoints_to_images_net([
        observed_keypoints,
        input_images[:, 0, Ellipsis],
        observed_keypoints[:, 0, Ellipsis]
    ])

    # Classifier head (bag_by_video: batch dimension == num_chunks)
    labels = tf.keras.Input(shape=(), dtype=tf.int32, name="label")

    # kp_flat = tf.keras.layers.TimeDistributed(
    #     tf.keras.layers.Flatten(), name="kp_flat"
    # )(observed_keypoints)  # (num_chunks, T, F)

    K = cfg.num_keypoints
    # observed_keypoints_stop = tf.keras.layers.Lambda(tf.stop_gradient)(
    #   observed_keypoints)

    kp_video = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [1, -1, K, 3]),name="kp_video")(observed_keypoints)

    dynamics_model = dynamics.build_vrnn(cfg)
    dyn_T = cfg.observed_steps + cfg.predicted_steps
    n16 = tf.shape(kp_video)[1] // dyn_T
    kp16 = tf.reshape(kp_video[:, :n16*dyn_T, ...], [n16, dyn_T, K, 3])  # [n16,16,K,3]

    pred16, kl_divergence = dynamics_model(kp16)
    mixed = tf.concat(
    [kp16[:, :cfg.observed_steps, :, :],          # keep original observed
     pred16[:, cfg.observed_steps:, :, :]],       # keep VRNN predicted steps
    axis=1)
    kp_video_pred = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [1, -1, K, 3]),name="kp_video_pred")(mixed)
    # kp_video_pred_stop = tf.keras.layers.Lambda(tf.stop_gradient)(
    #   kp_video_pred)
    kp_video_interp = tf.keras.layers.Lambda(lambda x: linear_interpolate_keypoints_xy(x, mu_thresh=0.05, keep_mu=True),
    name="kp_interp")(kp_video_pred)
    kp_video_norm = tf.keras.layers.Lambda(lambda x: normalize_kp_video(x, eps=1e-6, center_over_keypoints=True),
    name="kp_norm")(kp_video_interp)
    kp_long = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(), name="kp_flat_video")(kp_video_norm)

    # concatenate chunks into one long timeline for ONE video -> (1, total_T, F)
    # kp_long = tf.keras.layers.Lambda(
    #     lambda x: tf.reshape(x, [1, -1, F]),
    #     name="kp_long"
    # )(kp_flat)

    h = tf.keras.layers.LSTM(128, name="cls_lstm")(kp_long)
    h = tf.keras.layers.Dropout(0.2)(h)
    logits_video = tf.keras.layers.Dense(2, name="cls_logits")(h)

    model = tf.keras.Model(
        inputs=[input_images, labels],
        outputs=[observed_keypoints, logits_video],
        name="kp_cls_kfold"
    )

    # ----- Losses -----
    image_loss = tf.nn.l2_loss(input_images - reconstructed_images)
    image_loss /= tf.to_float(tf.shape(input_images)[0] * tf.shape(input_images)[1])
    model.add_loss(image_loss)

    separation_loss = losses.temporal_separation_loss(cfg, observed_keypoints[:, :, Ellipsis])
    model.add_loss(cfg.separation_loss_scale * separation_loss)

    sparsity_loss = tf.reduce_mean(tf.abs(mu))
    model.add_loss(cfg.sparsity_loss_scale * sparsity_loss)
    vrnn_coord_pred_loss = tf.nn.l2_loss(kp16 - pred16)

  # Normalize by batch size and sequence length:
    vrnn_coord_pred_loss /= tf.to_float(
      tf.shape(input_images)[0] * tf.shape(input_images)[1])
    model.add_loss(vrnn_coord_pred_loss)

    kl_loss = tf.reduce_mean(kl_divergence)  # Mean over batch and timesteps.
    model.add_loss(cfg.kl_loss_scale * kl_loss)

    label_video = labels[0:1]  # (1,)
    cls_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_video,
            logits=logits_video
        )
    )
    model.add_loss(cfg.cls_loss_scale * cls_loss)

    pred_video = tf.argmax(logits_video, axis=-1, output_type=tf.int32)  # (1,)
    acc_video = tf.reduce_mean(tf.cast(tf.equal(pred_video, label_video), tf.float32))
    model.add_metric(acc_video, name="cls_acc_video", aggregation="mean")
    model.add_metric(cls_loss, name="cls_loss_video", aggregation="mean")

    return model


# ---------------- Main: 5-fold CV on train folder, fixed test folder ----------------
def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    cfg = hyperparameters.get_config()

    # You can control these from cfg too if you want
    cfg.seed = getattr(cfg, "seed", 0)
    K_FOLDS = 5

    train_dir = os.path.join(cfg.data_dir, cfg.train_dir)
    test_dir  = os.path.join(cfg.data_dir, cfg.test_dir)

    all_train_files = list_npz_files(train_dir)  # your 50 (30 ADL + 20 Fall)
    all_test_files  = list_npz_files(test_dir)   # fixed separate test set

    print("Train folder videos:", len(all_train_files))
    print("Test  folder videos:", len(all_test_files))

    # Make a fixed test dataset once (no shuffling/repeat)
    test_ds, _ = make_bag_dataset(
        cfg,
        all_test_files,
        repeat_dataset=False,
        shuffle_files=False,
        shuffle_bags=False
    )

    # fold_test_accs = []
    # fold_val_accs = []
    fold_test = []
    fold_val = []

    for fold_i, (tr_files, va_files) in enumerate(stratified_kfold_identity(all_train_files, k=K_FOLDS, seed=cfg.seed)):
        print("\n==================== FOLD {}/{} ====================".format(fold_i + 1, K_FOLDS))
        print("Train videos:", len(tr_files), "| Val videos:", len(va_files))

        # Make each fold repeatable but different across folds
        #set_all_seeds(cfg.seed + fold_i)
        set_all_seeds(0)

        # IMPORTANT: reset graph state between folds
        tf.keras.backend.clear_session()

        # datasets for this fold
        train_ds, data_shapes = make_bag_dataset(
            cfg, tr_files,
            repeat_dataset=True,
            shuffle_files=True,
            shuffle_bags=True
        )
        val_ds, _ = make_bag_dataset(
            cfg, va_files,
            repeat_dataset=False,
            shuffle_files=False,
            shuffle_bags=False
        )

        model = build_model(cfg, data_shapes)

        optimizer = tf.keras.optimizers.Adam(
            lr=cfg.learning_rate, clipnorm=cfg.clipnorm
        )
        model.compile(optimizer=optimizer)

        # In bag_by_video=True, 1 step ~= 1 video bag
        steps_per_epoch = 100 #len(tr_files)
        val_steps = len(va_files)

        # checkpoint per fold
        os.makedirs("trained_model", exist_ok=True)
        ckpt_path = "trained_model/URFall_aug_occlusion_vrnn_trial_kfold_fold{}_T{}_bestf1_seed0_new.ckpt".format(
            fold_i + 1, 24 #cfg.observed_steps + cfg.predicted_steps
        )
        # ckpt_path = "trained_model/NewFall_subjectwise_fold{}_val{}_T{}_bestf1_seed0.ckpt".format(
        #     fold_i + 1,
        #     val_subject,
        #     24
        # )

        # Save best by validation video accuracy
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=ckpt_path,
        #     save_weights_only=True,
        #     monitor="val_cls_acc_video",
        #     mode="max",
        #     save_best_only=True,
        #     verbose=1
        # )
        save_last_best = SaveLastBest(
            filepath=ckpt_path,
            monitor="val_cls_acc_video",
            mode="max",
            atol=1e-8,
            verbose=1
        )

        model.fit(
            x=train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=cfg.num_epochs,
            validation_data=val_ds,
            validation_steps=val_steps,
            callbacks=[save_last_best],
            verbose=1
        )

        # Load best checkpoint for evaluation
        model.load_weights(ckpt_path)
        print("Loaded best:", ckpt_path)

        # Evaluate on val fold (video-level)
        # val_acc, val_cm = eval_video_level(model, val_ds, num_videos=len(va_files))
        # print("Fold VAL acc:", val_acc)
        # print("Fold VAL confusion matrix:\n", val_cm)

        # # Evaluate on fixed test set (video-level)
        # test_acc, test_cm = eval_video_level(model, test_ds, num_videos=len(all_test_files))
        # print("Fold TEST acc:", test_acc)
        # print("Fold TEST confusion matrix:\n", test_cm)
        # =========================
        # 1) Pick best threshold on VAL (maximize F1)
        # =========================
        yv, pv, _ = collect_probs_video_level(model, val_ds, n_videos=len(va_files))
        best_thr, (va_acc, va_prec, va_rec, va_f1, va_cm) = find_best_threshold_by_f1(yv, pv, num_thresholds=1001)

        print("\n[FOLD VAL] (threshold tuned on VAL for max F1)")
        print(f"Best threshold (P(Fall) >= thr): thr={best_thr:.4f}")
        print(f"Acc={va_acc:.3f}  Prec={va_prec:.3f}  Rec={va_rec:.3f}  F1={va_f1:.3f}")
        print("Confusion (rows true [ADL,Fall], cols pred [ADL,Fall]):\n", va_cm)

        # =========================
        # 2) Apply that threshold to FIXED TEST
        # =========================
        yt, pt, _ = collect_probs_video_level(model, test_ds, n_videos=len(all_test_files))
        te_acc, te_prec, te_rec, te_f1, te_cm = metrics_from_probs(yt, pt, threshold=best_thr)

        print("\n[FOLD TEST] (fixed test set, using VAL-tuned threshold)")
        print(f"thr={best_thr:.4f} | Acc={te_acc:.3f}  Prec={te_prec:.3f}  Rec={te_rec:.3f}  F1={te_f1:.3f}")
        print("Confusion (rows true [ADL,Fall], cols pred [ADL,Fall]):\n", te_cm)

        fold_val.append([va_acc, va_prec, va_rec, va_f1, best_thr])
        fold_test.append([te_acc, te_prec, te_rec, te_f1, best_thr])

    #     fold_val_accs.append(val_acc)
    #     fold_test_accs.append(test_acc)

    # # Summary
    # print("\n==================== SUMMARY ====================")
    # fold_val_accs = np.array(fold_val_accs, dtype=np.float32)
    # fold_test_accs = np.array(fold_test_accs, dtype=np.float32)
    # print("VAL  acc mean/std:", float(fold_val_accs.mean()), float(fold_val_accs.std()))
    # print("TEST acc mean/std:", float(fold_test_accs.mean()), float(fold_test_accs.std()))
    fold_val = np.array(fold_val, dtype=np.float32)
    fold_test = np.array(fold_test, dtype=np.float32)

    print("\n==================== SUMMARY (mean ± std across folds) ====================")
    for name, arr in [("VAL", fold_val), ("TEST", fold_test)]:
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        print(f"{name}: Acc {mean[0]:.3f}±{std[0]:.3f} | Prec {mean[1]:.3f}±{std[1]:.3f} | "
              f"Rec {mean[2]:.3f}±{std[2]:.3f} | F1 {mean[3]:.3f}±{std[3]:.3f}")


if __name__ == "__main__":
    app.run(main)