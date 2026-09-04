# coding=utf-8
"""Load sequence datasets into tf.data.Dataset pipeline (with optional video-level bags)."""

import functools
import os
import numpy as np
import tensorflow.compat.v1 as tf

# Data fields used by the model:
REQUIRED_DATA_FIELDS = ['image', 'true_object_pos']


def get_sequence_dataset(data_dir,
                         batch_size,
                         num_timesteps,
                         file_glob='*.npz',
                         random_offset=True,
                         repeat_dataset=True,
                         seed=0,
                         # labeling
                         num_adl=None,
                         num_fall=None,
                         # bag/mil
                         bag_by_video=False,
                         max_chunks_per_video=25,
                         # NEW (for k-fold)
                         filenames_override=None,
                         label_map_override=None,
                         shuffle_files=True,
                         shuffle_bags=True):
  """
  If bag_by_video=False:
    returns chunk-level dataset (batched by batch_size)

  If bag_by_video=True:
    returns one element per video:
      image: [num_chunks, num_timesteps, H, W, C]
      label: [num_chunks]
    No batching (num_chunks varies per video).
    To avoid OOM, we cap to max_chunks_per_video.
  """

  # ---------- filenames ----------
  if filenames_override is not None:
    filenames = list(filenames_override)  # expect full paths
  else:
    file_glob = file_glob if '.npz' in file_glob else file_glob + '.npz'
    filenames = sorted(tf.io.gfile.glob(os.path.join(data_dir, file_glob)))

  if not filenames:
    raise RuntimeError('No data files match {}'.format(
        os.path.join(data_dir, file_glob) if filenames_override is None else "filenames_override"))

  # ---------- label_map ----------
  def _label_from_name(path):
    b = os.path.basename(path).lower()
    if b.startswith("adl-"):
      return np.int32(0)
    if b.startswith("fall-"):
      return np.int32(1)
    raise ValueError("Unknown label for file: {}".format(b))

  if label_map_override is not None:
    label_map = dict(label_map_override)
  else:
    label_map = None
    if (num_adl is not None) or (num_fall is not None):
      # your old behavior (kept for backward compatibility)
      if num_adl is None or num_fall is None:
        raise ValueError("Provide both num_adl and num_fall, or neither.")
      expected = num_adl + num_fall
      if len(filenames) != expected:
        raise RuntimeError(
            "Expected {} files in {}, got {}."
            .format(expected, data_dir, len(filenames))
        )
      labels = [0] * num_adl + [1] * num_fall
      label_map = {os.path.basename(f): np.int32(labels[i]) for i, f in enumerate(filenames)}
    else:
      # NEW default for k-fold: label by filename prefix
      label_map = {os.path.basename(f): _label_from_name(f) for f in filenames}

  # ---------- shuffle file order (optional) ----------
  if shuffle_files:
    np.random.RandomState(seed).shuffle(filenames)

  # Create dataset from generator (one element = one full video sequence)
  dtypes, pre_chunk_shapes = _read_data_types_and_shapes(filenames, label_map)
  output_signature = {
      key: tf.TensorSpec(shape=pre_chunk_shapes[key], dtype=dtypes[key])
      for key in dtypes
  }
  dataset = tf.data.Dataset.from_generator(
      lambda: _read_numpy_sequences(filenames, label_map),
      output_signature=output_signature)

  if repeat_dataset:
    dataset = dataset.repeat()

  # Chunk each full video sequence into [num_chunks, num_timesteps, ...]
  chunk_fn = functools.partial(
      _chunk_sequence,
      chunk_length=num_timesteps,
      random_offset=random_offset,
      bag_by_video=bag_by_video,
      seed=seed
  )

  cycle = 1 if bag_by_video else batch_size
  dataset = dataset.interleave(chunk_fn, cycle_length=cycle)

  # Collapse labels/filenames after chunking (so label is per-chunk, not per-frame)
  if label_map is not None:
    if bag_by_video:
      def _collapse_label(d):
        out = dict(d)
        out['label'] = out['label'][:, 0]
        out['filename'] = out['filename'][:, 0]
        out['frame_ind'] = out['frame_ind'][:, 0]
        return out
      dataset = dataset.map(_collapse_label, num_parallel_calls=None)
    else:
      def _collapse_label(d):
        out = dict(d)
        out['label'] = out['label'][0]
        out['filename'] = out['filename'][0]
        out['frame_ind'] = out['frame_ind'][0]
        return out
      dataset = dataset.map(_collapse_label, num_parallel_calls=None)

  # ---- IMPORTANT: cap bag size to avoid OOM ----
  if bag_by_video:
    if max_chunks_per_video is None or max_chunks_per_video <= 0:
      raise ValueError("max_chunks_per_video must be a positive int when bag_by_video=True")

    def _cap_bag(d):
      out = dict(d)
      n = tf.shape(out['image'])[0]
      m = tf.minimum(n, tf.constant(max_chunks_per_video, tf.int32))
      n_minus_1 = tf.maximum(n - 1, 0)
      idx = tf.cast(tf.linspace(0.0, tf.cast(n_minus_1, tf.float32), m), tf.int32)
      for k, v in out.items():
        out[k] = tf.gather(v, idx, axis=0)
      return out

    dataset = dataset.map(_cap_bag, num_parallel_calls=None)

    # Shuffle across videos (bags) AFTER capping (optional)
    if shuffle_bags:
      dataset = dataset.shuffle(50, seed=seed, reshuffle_each_iteration=True)

  else:
    dataset = dataset.shuffle(100 * batch_size, seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)

  dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

  # Shapes output (used by build_model)
  def format_shape(spec):
    return (None,) + tuple(spec.shape.as_list()[1:])

  shapes = {key: format_shape(spec) for key, spec in dataset.element_spec.items()}

  return dataset, shapes


def _read_numpy_sequences(filenames, label_map=None):
  """Generator that reads Numpy files from disk into a dict."""
  for filename in filenames:
    try:
      with tf.io.gfile.GFile(filename, 'rb') as f:
        sequence_dict = {k: v for k, v in np.load(f).items()}
    except IOError as e:
      print('Caught IOError: "{}". Skipping file {}.'.format(e, filename))
      continue

    # Keep only required fields
    sequence_dict = _choose_data_fields(sequence_dict)

    # Adjust precision
    sequence_dict = {k: _adjust_precision_for_tf(v) for k, v in sequence_dict.items()}

    # Format image: uint8 -> float32 [-0.5, 0.5]
    sequence_dict['image'] = _format_image_data(sequence_dict['image'])

    num_frames = sequence_dict['image'].shape[0]

    # Add label per frame (so chunking can reshape it)
    if label_map is not None:
      base = os.path.basename(filename)
      y = label_map[base]
      sequence_dict['label'] = np.full((num_frames,), y, dtype=np.int32)

    # Traceability fields (per frame)
    sequence_dict['frame_ind'] = np.arange(num_frames, dtype=np.int32)
    sequence_dict['filename'] = np.full((num_frames,), os.path.basename(filename))

    yield sequence_dict


def _choose_data_fields(data_dict):
  """Returns a new dict containing only fields required by the model."""
  output_dict = {}
  for k in REQUIRED_DATA_FIELDS:
    if k in data_dict:
      output_dict[k] = data_dict[k]
    elif k == 'true_object_pos':
      print('Found no true_object_pos in data, adding dummy.')
      num_timesteps = data_dict['image'].shape[0]
      output_dict['true_object_pos'] = np.zeros([num_timesteps, 0, 2])
    else:
      raise ValueError(
          'Required key "{}" is not in the dict with keys {}.'.format(
              k, list(data_dict.keys())))
  return output_dict


def _adjust_precision_for_tf(array):
  """Adjusts precision for TensorFlow."""
  if array.dtype == np.float64:
    return array.astype(np.float32)
  if array.dtype == np.int64:
    return array.astype(np.int32)
  return array


def _format_image_data(image):
  """Formats uint8 input image to float32 in range [-0.5, 0.5]."""
  if not np.issubdtype(image.dtype, np.uint8):
    raise ValueError('Expected image uint8 but got {}.'.format(image.dtype))
  return image.astype(np.float32) / 255.0 - 0.5


def _read_data_types_and_shapes(filenames, label_map=None):
  """Gets dtypes and shapes for all keys in the dataset."""
  sequences = _read_numpy_sequences(filenames, label_map)
  sequence = next(sequences)
  sequences.close()
  dtypes = {k: tf.as_dtype(v.dtype) for k, v in sequence.items()}
  shapes = {k: (None,) + v.shape[1:] for k, v in sequence.items()}
  return dtypes, shapes


def _chunk_sequence(sequence_dict, chunk_length, random_offset=False, bag_by_video=False, seed=0):
  """Splits ONE video sequence dict into chunks.

  If bag_by_video=False:
    returns Dataset of elements (each element is ONE chunk)

  If bag_by_video=True:
    returns Dataset with a SINGLE element (the whole bag of chunks)
  """
  length = tf.shape(list(sequence_dict.values())[0])[0]

  if random_offset:
    num_chunks = tf.maximum(1, length // chunk_length - 1)
    output_length = num_chunks * chunk_length
    max_offset = length - output_length
    offset = tf.random.uniform((), 0, max_offset + 1, dtype=tf.int32)
  else:
    num_chunks = length // chunk_length
    output_length = num_chunks * chunk_length
    #offset = 0
    offset = tf.maximum(length - output_length, 0)

  chunked = {}
  for key, tensor in sequence_dict.items():
    tensor = tensor[offset:offset + output_length]
    chunked_shape = [num_chunks, chunk_length] + tensor.shape[1:].as_list()
    chunked[key] = tf.reshape(tensor, chunked_shape)

  if bag_by_video:
    # ONE element per video: the entire bag
    return tf.data.Dataset.from_tensors(chunked)
  else:
    # MANY elements per video: one per chunk (original behavior)
    return tf.data.Dataset.from_tensor_slices(chunked).shuffle(
        buffer_size=tf.cast(length, tf.int64), seed=seed)
