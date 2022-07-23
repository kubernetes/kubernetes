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

"""Tests for benchmark logger."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tempfile
import time
import unittest

import mock
from absl.testing import flagsaver
import tensorflow as tf  # pylint: disable=g-bad-import-order

try:
  from google.cloud import bigquery
except ImportError:
  bigquery = None

from official.utils.flags import core as flags_core
from official.utils.logs import logger


class BenchmarkLoggerTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):  # pylint: disable=invalid-name
    super(BenchmarkLoggerTest, cls).setUpClass()
    flags_core.define_benchmark()

  def test_get_default_benchmark_logger(self):
    with flagsaver.flagsaver(benchmark_logger_type='foo'):
      self.assertIsInstance(logger.get_benchmark_logger(),
                            logger.BaseBenchmarkLogger)

  def test_config_base_benchmark_logger(self):
    with flagsaver.flagsaver(benchmark_logger_type='BaseBenchmarkLogger'):
      logger.config_benchmark_logger()
      self.assertIsInstance(logger.get_benchmark_logger(),
                            logger.BaseBenchmarkLogger)

  def test_config_benchmark_file_logger(self):
    # Set the benchmark_log_dir first since the benchmark_logger_type will need
    # the value to be set when it does the validation.
    with flagsaver.flagsaver(benchmark_log_dir='/tmp'):
      with flagsaver.flagsaver(benchmark_logger_type='BenchmarkFileLogger'):
        logger.config_benchmark_logger()
        self.assertIsInstance(logger.get_benchmark_logger(),
                              logger.BenchmarkFileLogger)

  @unittest.skipIf(bigquery is None, 'Bigquery dependency is not installed.')
  def test_config_benchmark_bigquery_logger(self):
    with flagsaver.flagsaver(benchmark_logger_type='BenchmarkBigQueryLogger'):
      logger.config_benchmark_logger()
      self.assertIsInstance(logger.get_benchmark_logger(),
                            logger.BenchmarkBigQueryLogger)


class BaseBenchmarkLoggerTest(tf.test.TestCase):

  def setUp(self):
    super(BaseBenchmarkLoggerTest, self).setUp()
    self._actual_log = tf.logging.info
    self.logged_message = None

    def mock_log(*args, **kwargs):
      self.logged_message = args
      self._actual_log(*args, **kwargs)

    tf.logging.info = mock_log

  def tearDown(self):
    super(BaseBenchmarkLoggerTest, self).tearDown()
    tf.logging.info = self._actual_log

  def test_log_metric(self):
    log = logger.BaseBenchmarkLogger()
    log.log_metric("accuracy", 0.999, global_step=1e4, extras={"name": "value"})

    expected_log_prefix = "Benchmark metric:"
    self.assertRegexpMatches(str(self.logged_message), expected_log_prefix)


class BenchmarkFileLoggerTest(tf.test.TestCase):

  def setUp(self):
    super(BenchmarkFileLoggerTest, self).setUp()
    # Avoid pulling extra env vars from test environment which affects the test
    # result, eg. Kokoro test has a TF_PKG env which affect the test case
    # test_collect_tensorflow_environment_variables()
    self.original_environ = dict(os.environ)
    os.environ.clear()

  def tearDown(self):
    super(BenchmarkFileLoggerTest, self).tearDown()
    tf.gfile.DeleteRecursively(self.get_temp_dir())
    os.environ.clear()
    os.environ.update(self.original_environ)

  def test_create_logging_dir(self):
    non_exist_temp_dir = os.path.join(self.get_temp_dir(), "unknown_dir")
    self.assertFalse(tf.gfile.IsDirectory(non_exist_temp_dir))

    logger.BenchmarkFileLogger(non_exist_temp_dir)
    self.assertTrue(tf.gfile.IsDirectory(non_exist_temp_dir))

  def test_log_metric(self):
    log_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    log = logger.BenchmarkFileLogger(log_dir)
    log.log_metric("accuracy", 0.999, global_step=1e4, extras={"name": "value"})

    metric_log = os.path.join(log_dir, "metric.log")
    self.assertTrue(tf.gfile.Exists(metric_log))
    with tf.gfile.GFile(metric_log) as f:
      metric = json.loads(f.readline())
      self.assertEqual(metric["name"], "accuracy")
      self.assertEqual(metric["value"], 0.999)
      self.assertEqual(metric["unit"], None)
      self.assertEqual(metric["global_step"], 1e4)
      self.assertEqual(metric["extras"], [{"name": "name", "value": "value"}])

  def test_log_multiple_metrics(self):
    log_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    log = logger.BenchmarkFileLogger(log_dir)
    log.log_metric("accuracy", 0.999, global_step=1e4, extras={"name": "value"})
    log.log_metric("loss", 0.02, global_step=1e4)

    metric_log = os.path.join(log_dir, "metric.log")
    self.assertTrue(tf.gfile.Exists(metric_log))
    with tf.gfile.GFile(metric_log) as f:
      accuracy = json.loads(f.readline())
      self.assertEqual(accuracy["name"], "accuracy")
      self.assertEqual(accuracy["value"], 0.999)
      self.assertEqual(accuracy["unit"], None)
      self.assertEqual(accuracy["global_step"], 1e4)
      self.assertEqual(accuracy["extras"], [{"name": "name", "value": "value"}])

      loss = json.loads(f.readline())
      self.assertEqual(loss["name"], "loss")
      self.assertEqual(loss["value"], 0.02)
      self.assertEqual(loss["unit"], None)
      self.assertEqual(loss["global_step"], 1e4)
      self.assertEqual(loss["extras"], [])

  def test_log_non_number_value(self):
    log_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    log = logger.BenchmarkFileLogger(log_dir)
    const = tf.constant(1)
    log.log_metric("accuracy", const)

    metric_log = os.path.join(log_dir, "metric.log")
    self.assertFalse(tf.gfile.Exists(metric_log))

  def test_log_evaluation_result(self):
    eval_result = {"loss": 0.46237424,
                   "global_step": 207082,
                   "accuracy": 0.9285}
    log_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    log = logger.BenchmarkFileLogger(log_dir)
    log.log_evaluation_result(eval_result)

    metric_log = os.path.join(log_dir, "metric.log")
    self.assertTrue(tf.gfile.Exists(metric_log))
    with tf.gfile.GFile(metric_log) as f:
      accuracy = json.loads(f.readline())
      self.assertEqual(accuracy["name"], "accuracy")
      self.assertEqual(accuracy["value"], 0.9285)
      self.assertEqual(accuracy["unit"], None)
      self.assertEqual(accuracy["global_step"], 207082)

      loss = json.loads(f.readline())
      self.assertEqual(loss["name"], "loss")
      self.assertEqual(loss["value"], 0.46237424)
      self.assertEqual(loss["unit"], None)
      self.assertEqual(loss["global_step"], 207082)

  def test_log_evaluation_result_with_invalid_type(self):
    eval_result = "{'loss': 0.46237424, 'global_step': 207082}"
    log_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    log = logger.BenchmarkFileLogger(log_dir)
    log.log_evaluation_result(eval_result)

    metric_log = os.path.join(log_dir, "metric.log")
    self.assertFalse(tf.gfile.Exists(metric_log))

  def test_collect_tensorflow_info(self):
    run_info = {}
    logger._collect_tensorflow_info(run_info)
    self.assertNotEqual(run_info["tensorflow_version"], {})
    self.assertEqual(run_info["tensorflow_version"]["version"], tf.VERSION)
    self.assertEqual(run_info["tensorflow_version"]["git_hash"], tf.GIT_VERSION)

  def test_collect_run_params(self):
    run_info = {}
    run_parameters = {
        "batch_size": 32,
        "synthetic_data": True,
        "train_epochs": 100.00,
        "dtype": "fp16",
        "resnet_size": 50,
        "random_tensor": tf.constant(2.0)
    }
    logger._collect_run_params(run_info, run_parameters)
    self.assertEqual(len(run_info["run_parameters"]), 6)
    self.assertEqual(run_info["run_parameters"][0],
                     {"name": "batch_size", "long_value": 32})
    self.assertEqual(run_info["run_parameters"][1],
                     {"name": "dtype", "string_value": "fp16"})
    self.assertEqual(run_info["run_parameters"][2],
                     {"name": "random_tensor", "string_value":
                          "Tensor(\"Const:0\", shape=(), dtype=float32)"})
    self.assertEqual(run_info["run_parameters"][3],
                     {"name": "resnet_size", "long_value": 50})
    self.assertEqual(run_info["run_parameters"][4],
                     {"name": "synthetic_data", "bool_value": "True"})
    self.assertEqual(run_info["run_parameters"][5],
                     {"name": "train_epochs", "float_value": 100.00})

  def test_collect_tensorflow_environment_variables(self):
    os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"
    os.environ["TF_OTHER"] = "2"
    os.environ["OTHER"] = "3"

    run_info = {}
    logger._collect_tensorflow_environment_variables(run_info)
    self.assertIsNotNone(run_info["tensorflow_environment_variables"])
    expected_tf_envs = [
        {"name": "TF_ENABLE_WINOGRAD_NONFUSED", "value": "1"},
        {"name": "TF_OTHER", "value": "2"},
    ]
    self.assertEqual(run_info["tensorflow_environment_variables"],
                     expected_tf_envs)

  @unittest.skipUnless(tf.test.is_built_with_cuda(), "requires GPU")
  def test_collect_gpu_info(self):
    run_info = {"machine_config": {}}
    logger._collect_gpu_info(run_info)
    self.assertNotEqual(run_info["machine_config"]["gpu_info"], {})

  def test_collect_memory_info(self):
    run_info = {"machine_config": {}}
    logger._collect_memory_info(run_info)
    self.assertIsNotNone(run_info["machine_config"]["memory_total"])
    self.assertIsNotNone(run_info["machine_config"]["memory_available"])


@unittest.skipIf(bigquery is None, 'Bigquery dependency is not installed.')
class BenchmarkBigQueryLoggerTest(tf.test.TestCase):

  def setUp(self):
    super(BenchmarkBigQueryLoggerTest, self).setUp()
    # Avoid pulling extra env vars from test environment which affects the test
    # result, eg. Kokoro test has a TF_PKG env which affect the test case
    # test_collect_tensorflow_environment_variables()
    self.original_environ = dict(os.environ)
    os.environ.clear()

    self.mock_bq_uploader = mock.MagicMock()
    self.logger = logger.BenchmarkBigQueryLogger(
        self.mock_bq_uploader, "dataset", "run_table", "metric_table",
        "run_id")

  def tearDown(self):
    super(BenchmarkBigQueryLoggerTest, self).tearDown()
    tf.gfile.DeleteRecursively(self.get_temp_dir())
    os.environ.clear()
    os.environ.update(self.original_environ)

  def test_log_metric(self):
    self.logger.log_metric(
        "accuracy", 0.999, global_step=1e4, extras={"name": "value"})
    expected_metric_json = [{
        "name": "accuracy",
        "value": 0.999,
        "unit": None,
        "global_step": 1e4,
        "timestamp": mock.ANY,
        "extras": [{"name": "name", "value": "value"}]
    }]
    # log_metric will call upload_benchmark_metric_json in a separate thread.
    # Give it some grace period for the new thread before assert.
    time.sleep(1)
    self.mock_bq_uploader.upload_benchmark_metric_json.assert_called_once_with(
        "dataset", "metric_table", "run_id", expected_metric_json)


if __name__ == "__main__":
  tf.test.main()
