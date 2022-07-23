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
"""Flags which will be nearly universal across models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf

from official.utils.flags._conventions import help_wrap
from official.utils.logs import hooks_helper


def define_base(data_dir=True, model_dir=True, train_epochs=True,
                epochs_between_evals=True, stop_threshold=True, batch_size=True,
                multi_gpu=False, num_gpu=True, hooks=True, export_dir=True):
  """Register base flags.

  Args:
    data_dir: Create a flag for specifying the input data directory.
    model_dir: Create a flag for specifying the model file directory.
    train_epochs: Create a flag to specify the number of training epochs.
    epochs_between_evals: Create a flag to specify the frequency of testing.
    stop_threshold: Create a flag to specify a threshold accuracy or other
      eval metric which should trigger the end of training.
    batch_size: Create a flag to specify the batch size.
    multi_gpu: Create a flag to allow the use of all available GPUs.
    num_gpu: Create a flag to specify the number of GPUs used.
    hooks: Create a flag to specify hooks for logging.
    export_dir: Create a flag to specify where a SavedModel should be exported.

  Returns:
    A list of flags for core.py to marks as key flags.
  """
  key_flags = []

  if data_dir:
    flags.DEFINE_string(
        name="data_dir", short_name="dd", default="/tmp",
        help=help_wrap("The location of the input data."))
    key_flags.append("data_dir")

  if model_dir:
    flags.DEFINE_string(
        name="model_dir", short_name="md", default="/tmp",
        help=help_wrap("The location of the model checkpoint files."))
    key_flags.append("model_dir")

  if train_epochs:
    flags.DEFINE_integer(
        name="train_epochs", short_name="te", default=1,
        help=help_wrap("The number of epochs used to train."))
    key_flags.append("train_epochs")

  if epochs_between_evals:
    flags.DEFINE_integer(
        name="epochs_between_evals", short_name="ebe", default=1,
        help=help_wrap("The number of training epochs to run between "
                       "evaluations."))
    key_flags.append("epochs_between_evals")

  if stop_threshold:
    flags.DEFINE_float(
        name="stop_threshold", short_name="st",
        default=None,
        help=help_wrap("If passed, training will stop at the earlier of "
                       "train_epochs and when the evaluation metric is  "
                       "greater than or equal to stop_threshold."))

  if batch_size:
    flags.DEFINE_integer(
        name="batch_size", short_name="bs", default=32,
        help=help_wrap("Batch size for training and evaluation."))
    key_flags.append("batch_size")

  assert not (multi_gpu and num_gpu)

  if multi_gpu:
    flags.DEFINE_bool(
        name="multi_gpu", default=False,
        help=help_wrap("If set, run across all available GPUs."))
    key_flags.append("multi_gpu")

  if num_gpu:
    flags.DEFINE_integer(
        name="num_gpus", short_name="ng",
        default=1 if tf.test.is_gpu_available() else 0,
        help=help_wrap(
            "How many GPUs to use with the DistributionStrategies API. The "
            "default is 1 if TensorFlow can detect a GPU, and 0 otherwise."))

  if hooks:
    # Construct a pretty summary of hooks.
    hook_list_str = (
        u"\ufeff  Hook:\n" + u"\n".join([u"\ufeff    {}".format(key) for key
                                         in hooks_helper.HOOKS]))
    flags.DEFINE_list(
        name="hooks", short_name="hk", default="LoggingTensorHook",
        help=help_wrap(
            u"A list of (case insensitive) strings to specify the names of "
            u"training hooks.\n{}\n\ufeff  Example: `--hooks ProfilerHook,"
            u"ExamplesPerSecondHook`\n See official.utils.logs.hooks_helper "
            u"for details.".format(hook_list_str))
    )
    key_flags.append("hooks")

  if export_dir:
    flags.DEFINE_string(
        name="export_dir", short_name="ed", default=None,
        help=help_wrap("If set, a SavedModel serialization of the model will "
                       "be exported to this directory at the end of training. "
                       "See the README for more details and relevant links.")
    )
    key_flags.append("export_dir")

  return key_flags


def get_num_gpus(flags_obj):
  """Treat num_gpus=-1 as 'use all'."""
  if flags_obj.num_gpus != -1:
    return flags_obj.num_gpus

  from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top
  local_device_protos = device_lib.list_local_devices()
  return sum([1 for d in local_device_protos if d.device_type == "GPU"])
