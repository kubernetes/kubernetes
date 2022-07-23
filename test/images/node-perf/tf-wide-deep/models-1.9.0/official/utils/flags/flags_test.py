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

import unittest

from absl import flags
import tensorflow as tf

from official.utils.flags import core as flags_core  # pylint: disable=g-bad-import-order


def define_flags():
  flags_core.define_base(multi_gpu=True, num_gpu=False)
  flags_core.define_performance()
  flags_core.define_image()
  flags_core.define_benchmark()


class BaseTester(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(BaseTester, cls).setUpClass()
    define_flags()

  def test_default_setting(self):
    """Test to ensure fields exist and defaults can be set.
    """

    defaults = dict(
        data_dir="dfgasf",
        model_dir="dfsdkjgbs",
        train_epochs=534,
        epochs_between_evals=15,
        batch_size=256,
        hooks=["LoggingTensorHook"],
        num_parallel_calls=18,
        inter_op_parallelism_threads=5,
        intra_op_parallelism_threads=10,
        data_format="channels_first"
    )

    flags_core.set_defaults(**defaults)
    flags_core.parse_flags()

    for key, value in defaults.items():
      assert flags.FLAGS.get_flag_value(name=key, default=None) == value

  def test_benchmark_setting(self):
    defaults = dict(
        hooks=["LoggingMetricHook"],
        benchmark_log_dir="/tmp/12345",
        gcp_project="project_abc",
    )

    flags_core.set_defaults(**defaults)
    flags_core.parse_flags()

    for key, value in defaults.items():
      assert flags.FLAGS.get_flag_value(name=key, default=None) == value

  def test_booleans(self):
    """Test to ensure boolean flags trigger as expected.
    """

    flags_core.parse_flags([__file__, "--multi_gpu", "--use_synthetic_data"])

    assert flags.FLAGS.multi_gpu
    assert flags.FLAGS.use_synthetic_data

  def test_parse_dtype_info(self):
    for dtype_str, tf_dtype, loss_scale in [["fp16", tf.float16, 128],
                                            ["fp32", tf.float32, 1]]:
      flags_core.parse_flags([__file__, "--dtype", dtype_str])

      self.assertEqual(flags_core.get_tf_dtype(flags.FLAGS), tf_dtype)
      self.assertEqual(flags_core.get_loss_scale(flags.FLAGS), loss_scale)

      flags_core.parse_flags(
          [__file__, "--dtype", dtype_str, "--loss_scale", "5"])

      self.assertEqual(flags_core.get_loss_scale(flags.FLAGS), 5)

    with self.assertRaises(SystemExit):
      flags_core.parse_flags([__file__, "--dtype", "int8"])


if __name__ == "__main__":
  unittest.main()
