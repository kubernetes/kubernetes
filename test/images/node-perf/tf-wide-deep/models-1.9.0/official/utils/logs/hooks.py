# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Hook that counts examples per second every N steps or seconds."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.logs import logger


class ExamplesPerSecondHook(tf.train.SessionRunHook):
  """Hook to print out examples per second.

  Total time is tracked and then divided by the total number of steps
  to get the average step time and then batch_size is used to determine
  the running average of examples per second. The examples per second for the
  most recent interval is also logged.
  """

  def __init__(self,
               batch_size,
               every_n_steps=None,
               every_n_secs=None,
               warm_steps=0,
               metric_logger=None):
    """Initializer for ExamplesPerSecondHook.

    Args:
      batch_size: Total batch size across all workers used to calculate
        examples/second from global time.
      every_n_steps: Log stats every n steps.
      every_n_secs: Log stats every n seconds. Exactly one of the
        `every_n_steps` or `every_n_secs` should be set.
      warm_steps: The number of steps to be skipped before logging and running
        average calculation. warm_steps steps refers to global steps across all
        workers, not on each worker
      metric_logger: instance of `BenchmarkLogger`, the benchmark logger that
          hook should use to write the log. If None, BaseBenchmarkLogger will
          be used.

    Raises:
      ValueError: if neither `every_n_steps` or `every_n_secs` is set, or
      both are set.
    """

    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError("exactly one of every_n_steps"
                       " and every_n_secs should be provided.")

    self._logger = metric_logger or logger.BaseBenchmarkLogger()

    self._timer = tf.train.SecondOrStepTimer(
        every_steps=every_n_steps, every_secs=every_n_secs)

    self._step_train_time = 0
    self._total_steps = 0
    self._batch_size = batch_size
    self._warm_steps = warm_steps

  def begin(self):
    """Called once before using the session to check global step."""
    self._global_step_tensor = tf.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use StepCounterHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    """Called before each call to run().

    Args:
      run_context: A SessionRunContext object.

    Returns:
      A SessionRunArgs object or None if never triggered.
    """
    return tf.train.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    """Called after each call to run().

    Args:
      run_context: A SessionRunContext object.
      run_values: A SessionRunValues object.
    """
    global_step = run_values.results

    if self._timer.should_trigger_for_step(
        global_step) and global_step > self._warm_steps:
      elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
          global_step)
      if elapsed_time is not None:
        self._step_train_time += elapsed_time
        self._total_steps += elapsed_steps

        # average examples per second is based on the total (accumulative)
        # training steps and training time so far
        average_examples_per_sec = self._batch_size * (
            self._total_steps / self._step_train_time)
        # current examples per second is based on the elapsed training steps
        # and training time per batch
        current_examples_per_sec = self._batch_size * (
            elapsed_steps / elapsed_time)

        self._logger.log_metric(
            "average_examples_per_sec", average_examples_per_sec,
            global_step=global_step)

        self._logger.log_metric(
            "current_examples_per_sec", current_examples_per_sec,
            global_step=global_step)
