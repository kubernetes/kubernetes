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
"""Flags for benchmarking models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

from official.utils.flags._conventions import help_wrap


def define_benchmark(benchmark_log_dir=True, bigquery_uploader=True):
  """Register benchmarking flags.

  Args:
    benchmark_log_dir: Create a flag to specify location for benchmark logging.
    bigquery_uploader: Create flags for uploading results to BigQuery.

  Returns:
    A list of flags for core.py to marks as key flags.
  """

  key_flags = []

  flags.DEFINE_enum(
      name="benchmark_logger_type", default="BaseBenchmarkLogger",
      enum_values=["BaseBenchmarkLogger", "BenchmarkFileLogger",
                   "BenchmarkBigQueryLogger"],
      help=help_wrap("The type of benchmark logger to use. Defaults to using "
                     "BaseBenchmarkLogger which logs to STDOUT. Different "
                     "loggers will require other flags to be able to work."))

  if benchmark_log_dir:
    flags.DEFINE_string(
        name="benchmark_log_dir", short_name="bld", default=None,
        help=help_wrap("The location of the benchmark logging.")
    )

  if bigquery_uploader:
    flags.DEFINE_string(
        name="gcp_project", short_name="gp", default=None,
        help=help_wrap(
            "The GCP project name where the benchmark will be uploaded."))

    flags.DEFINE_string(
        name="bigquery_data_set", short_name="bds", default="test_benchmark",
        help=help_wrap(
            "The Bigquery dataset name where the benchmark will be uploaded."))

    flags.DEFINE_string(
        name="bigquery_run_table", short_name="brt", default="benchmark_run",
        help=help_wrap("The Bigquery table name where the benchmark run "
                       "information will be uploaded."))

    flags.DEFINE_string(
        name="bigquery_metric_table", short_name="bmt",
        default="benchmark_metric",
        help=help_wrap("The Bigquery table name where the benchmark metric "
                       "information will be uploaded."))

  @flags.multi_flags_validator(
      ["benchmark_logger_type", "benchmark_log_dir"],
      message="--benchmark_logger_type=BenchmarkFileLogger will require "
              "--benchmark_log_dir being set")
  def _check_benchmark_log_dir(flags_dict):
    benchmark_logger_type = flags_dict["benchmark_logger_type"]
    if benchmark_logger_type == "BenchmarkFileLogger":
      return flags_dict["benchmark_log_dir"]
    return True

  return key_flags
