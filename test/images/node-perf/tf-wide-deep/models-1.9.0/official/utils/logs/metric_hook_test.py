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
"""Tests for metric_hook."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import time

import tensorflow as tf  # pylint: disable=g-bad-import-order
from tensorflow.python.training import monitored_session  # pylint: disable=g-bad-import-order

from official.utils.logs import metric_hook
from official.utils.testing import mock_lib


class LoggingMetricHookTest(tf.test.TestCase):
  """Tests for LoggingMetricHook."""

  def setUp(self):
    super(LoggingMetricHookTest, self).setUp()

    self._log_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    self._logger = mock_lib.MockBenchmarkLogger()

  def tearDown(self):
    super(LoggingMetricHookTest, self).tearDown()
    tf.gfile.DeleteRecursively(self.get_temp_dir())

  def test_illegal_args(self):
    with self.assertRaisesRegexp(ValueError, "nvalid every_n_iter"):
      metric_hook.LoggingMetricHook(tensors=["t"], every_n_iter=0)
    with self.assertRaisesRegexp(ValueError, "nvalid every_n_iter"):
      metric_hook.LoggingMetricHook(tensors=["t"], every_n_iter=-10)
    with self.assertRaisesRegexp(ValueError, "xactly one of"):
      metric_hook.LoggingMetricHook(
          tensors=["t"], every_n_iter=5, every_n_secs=5)
    with self.assertRaisesRegexp(ValueError, "xactly one of"):
      metric_hook.LoggingMetricHook(tensors=["t"])
    with self.assertRaisesRegexp(ValueError, "metric_logger"):
      metric_hook.LoggingMetricHook(tensors=["t"], every_n_iter=5)

  def test_print_at_end_only(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      tf.train.get_or_create_global_step()
      t = tf.constant(42.0, name="foo")
      train_op = tf.constant(3)
      hook = metric_hook.LoggingMetricHook(
          tensors=[t.name], at_end=True, metric_logger=self._logger)
      hook.begin()
      mon_sess = monitored_session._HookedSession(sess, [hook])  # pylint: disable=protected-access
      sess.run(tf.global_variables_initializer())

      for _ in range(3):
        mon_sess.run(train_op)
        self.assertEqual(self._logger.logged_metric, [])

      hook.end(sess)
      self.assertEqual(len(self._logger.logged_metric), 1)
      metric = self._logger.logged_metric[0]
      self.assertRegexpMatches(metric["name"], "foo")
      self.assertEqual(metric["value"], 42.0)
      self.assertEqual(metric["unit"], None)
      self.assertEqual(metric["global_step"], 0)

  def test_global_step_not_found(self):
    with tf.Graph().as_default():
      t = tf.constant(42.0, name="foo")
      hook = metric_hook.LoggingMetricHook(
          tensors=[t.name], at_end=True, metric_logger=self._logger)

      with self.assertRaisesRegexp(
          RuntimeError, "should be created to use LoggingMetricHook."):
        hook.begin()

  def test_log_tensors(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      tf.train.get_or_create_global_step()
      t1 = tf.constant(42.0, name="foo")
      t2 = tf.constant(43.0, name="bar")
      train_op = tf.constant(3)
      hook = metric_hook.LoggingMetricHook(
          tensors=[t1, t2], at_end=True, metric_logger=self._logger)
      hook.begin()
      mon_sess = monitored_session._HookedSession(sess, [hook])  # pylint: disable=protected-access
      sess.run(tf.global_variables_initializer())

      for _ in range(3):
        mon_sess.run(train_op)
        self.assertEqual(self._logger.logged_metric, [])

      hook.end(sess)
      self.assertEqual(len(self._logger.logged_metric), 2)
      metric1 = self._logger.logged_metric[0]
      self.assertRegexpMatches(str(metric1["name"]), "foo")
      self.assertEqual(metric1["value"], 42.0)
      self.assertEqual(metric1["unit"], None)
      self.assertEqual(metric1["global_step"], 0)

      metric2 = self._logger.logged_metric[1]
      self.assertRegexpMatches(str(metric2["name"]), "bar")
      self.assertEqual(metric2["value"], 43.0)
      self.assertEqual(metric2["unit"], None)
      self.assertEqual(metric2["global_step"], 0)

  def _validate_print_every_n_steps(self, sess, at_end):
    t = tf.constant(42.0, name="foo")

    train_op = tf.constant(3)
    hook = metric_hook.LoggingMetricHook(
        tensors=[t.name], every_n_iter=10, at_end=at_end,
        metric_logger=self._logger)
    hook.begin()
    mon_sess = monitored_session._HookedSession(sess, [hook])  # pylint: disable=protected-access
    sess.run(tf.global_variables_initializer())
    mon_sess.run(train_op)
    self.assertRegexpMatches(str(self._logger.logged_metric), t.name)
    for _ in range(3):
      self._logger.logged_metric = []
      for _ in range(9):
        mon_sess.run(train_op)
        # assertNotRegexpMatches is not supported by python 3.1 and later
        self.assertEqual(str(self._logger.logged_metric).find(t.name), -1)
      mon_sess.run(train_op)
      self.assertRegexpMatches(str(self._logger.logged_metric), t.name)

    # Add additional run to verify proper reset when called multiple times.
    self._logger.logged_metric = []
    mon_sess.run(train_op)
    # assertNotRegexpMatches is not supported by python 3.1 and later
    self.assertEqual(str(self._logger.logged_metric).find(t.name), -1)

    self._logger.logged_metric = []
    hook.end(sess)
    if at_end:
      self.assertRegexpMatches(str(self._logger.logged_metric), t.name)
    else:
      # assertNotRegexpMatches is not supported by python 3.1 and later
      self.assertEqual(str(self._logger.logged_metric).find(t.name), -1)

  def test_print_every_n_steps(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      tf.train.get_or_create_global_step()
      self._validate_print_every_n_steps(sess, at_end=False)
      # Verify proper reset.
      self._validate_print_every_n_steps(sess, at_end=False)

  def test_print_every_n_steps_and_end(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      tf.train.get_or_create_global_step()
      self._validate_print_every_n_steps(sess, at_end=True)
      # Verify proper reset.
      self._validate_print_every_n_steps(sess, at_end=True)

  def _validate_print_every_n_secs(self, sess, at_end):
    t = tf.constant(42.0, name="foo")
    train_op = tf.constant(3)

    hook = metric_hook.LoggingMetricHook(
        tensors=[t.name], every_n_secs=1.0, at_end=at_end,
        metric_logger=self._logger)
    hook.begin()
    mon_sess = monitored_session._HookedSession(sess, [hook])  # pylint: disable=protected-access
    sess.run(tf.global_variables_initializer())

    mon_sess.run(train_op)
    self.assertRegexpMatches(str(self._logger.logged_metric), t.name)

    # assertNotRegexpMatches is not supported by python 3.1 and later
    self._logger.logged_metric = []
    mon_sess.run(train_op)
    self.assertEqual(str(self._logger.logged_metric).find(t.name), -1)
    time.sleep(1.0)

    self._logger.logged_metric = []
    mon_sess.run(train_op)
    self.assertRegexpMatches(str(self._logger.logged_metric), t.name)

    self._logger.logged_metric = []
    hook.end(sess)
    if at_end:
      self.assertRegexpMatches(str(self._logger.logged_metric), t.name)
    else:
      # assertNotRegexpMatches is not supported by python 3.1 and later
      self.assertEqual(str(self._logger.logged_metric).find(t.name), -1)

  def test_print_every_n_secs(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      tf.train.get_or_create_global_step()
      self._validate_print_every_n_secs(sess, at_end=False)
      # Verify proper reset.
      self._validate_print_every_n_secs(sess, at_end=False)

  def test_print_every_n_secs_and_end(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      tf.train.get_or_create_global_step()
      self._validate_print_every_n_secs(sess, at_end=True)
      # Verify proper reset.
      self._validate_print_every_n_secs(sess, at_end=True)


if __name__ == "__main__":
  tf.test.main()
