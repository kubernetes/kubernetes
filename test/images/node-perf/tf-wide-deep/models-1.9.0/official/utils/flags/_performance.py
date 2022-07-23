# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Register flags for optimizing performance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

from absl import flags    # pylint: disable=g-bad-import-order
import tensorflow as tf   # pylint: disable=g-bad-import-order

from official.utils.flags._conventions import help_wrap


# Map string to (TensorFlow dtype, default loss scale)
DTYPE_MAP = {
    "fp16": (tf.float16, 128),
    "fp32": (tf.float32, 1),
}


def get_tf_dtype(flags_obj):
  return DTYPE_MAP[flags_obj.dtype][0]


def get_loss_scale(flags_obj):
  if flags_obj.loss_scale is not None:
    return flags_obj.loss_scale
  return DTYPE_MAP[flags_obj.dtype][1]


def define_performance(num_parallel_calls=True, inter_op=True, intra_op=True,
                       synthetic_data=True, max_train_steps=True, dtype=True):
  """Register flags for specifying performance tuning arguments.

  Args:
    num_parallel_calls: Create a flag to specify parallelism of data loading.
    inter_op: Create a flag to allow specification of inter op threads.
    intra_op: Create a flag to allow specification of intra op threads.
    synthetic_data: Create a flag to allow the use of synthetic data.
    max_train_steps: Create a flags to allow specification of maximum number
      of training steps
    dtype: Create flags for specifying dtype.

  Returns:
    A list of flags for core.py to marks as key flags.
  """

  key_flags = []
  if num_parallel_calls:
    flags.DEFINE_integer(
        name="num_parallel_calls", short_name="npc",
        default=multiprocessing.cpu_count(),
        help=help_wrap("The number of records that are  processed in parallel "
                       "during input processing. This can be optimized per "
                       "data set but for generally homogeneous data sets, "
                       "should be approximately the number of available CPU "
                       "cores. (default behavior)"))

  if inter_op:
    flags.DEFINE_integer(
        name="inter_op_parallelism_threads", short_name="inter", default=0,
        help=help_wrap("Number of inter_op_parallelism_threads to use for CPU. "
                       "See TensorFlow config.proto for details.")
    )

  if intra_op:
    flags.DEFINE_integer(
        name="intra_op_parallelism_threads", short_name="intra", default=0,
        help=help_wrap("Number of intra_op_parallelism_threads to use for CPU. "
                       "See TensorFlow config.proto for details."))

  if synthetic_data:
    flags.DEFINE_bool(
        name="use_synthetic_data", short_name="synth", default=False,
        help=help_wrap(
            "If set, use fake data (zeroes) instead of a real dataset. "
            "This mode is useful for performance debugging, as it removes "
            "input processing steps, but will not learn anything."))

  if max_train_steps:
    flags.DEFINE_integer(
        name="max_train_steps", short_name="mts", default=None, help=help_wrap(
            "The model will stop training if the global_step reaches this "
            "value. If not set, training will run until the specified number "
            "of epochs have run as usual. It is generally recommended to set "
            "--train_epochs=1 when using this flag."
        ))

  if dtype:
    flags.DEFINE_enum(
        name="dtype", short_name="dt", default="fp32",
        enum_values=DTYPE_MAP.keys(),
        help=help_wrap("The TensorFlow datatype used for calculations. "
                       "Variables may be cast to a higher precision on a "
                       "case-by-case basis for numerical stability."))

    flags.DEFINE_integer(
        name="loss_scale", short_name="ls", default=None,
        help=help_wrap(
            "The amount to scale the loss by when the model is run. Before "
            "gradients are computed, the loss is multiplied by the loss scale, "
            "making all gradients loss_scale times larger. To adjust for this, "
            "gradients are divided by the loss scale before being applied to "
            "variables. This is mathematically equivalent to training without "
            "a loss scale, but the loss scale helps avoid some intermediate "
            "gradients from underflowing to zero. If not provided the default "
            "for fp16 is 128 and 1 for all other dtypes."))

    loss_scale_val_msg = "loss_scale should be a positive integer."
    @flags.validator(flag_name="loss_scale", message=loss_scale_val_msg)
    def _check_loss_scale(loss_scale):  # pylint: disable=unused-variable
      if loss_scale is None:
        return True  # null case is handled in get_loss_scale()

      return loss_scale > 0

  return key_flags
