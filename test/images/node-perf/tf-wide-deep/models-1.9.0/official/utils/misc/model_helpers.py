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
"""Miscellaneous functions that can be called by models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

import tensorflow as tf


def past_stop_threshold(stop_threshold, eval_metric):
  """Return a boolean representing whether a model should be stopped.

  Args:
    stop_threshold: float, the threshold above which a model should stop
      training.
    eval_metric: float, the current value of the relevant metric to check.

  Returns:
    True if training should stop, False otherwise.

  Raises:
    ValueError: if either stop_threshold or eval_metric is not a number
  """
  if stop_threshold is None:
    return False

  if not isinstance(stop_threshold, numbers.Number):
    raise ValueError("Threshold for checking stop conditions must be a number.")
  if not isinstance(eval_metric, numbers.Number):
    raise ValueError("Eval metric being checked against stop conditions "
                     "must be a number.")

  if eval_metric >= stop_threshold:
    tf.logging.info(
        "Stop threshold of {} was passed with metric value {}.".format(
            stop_threshold, eval_metric))
    return True

  return False
