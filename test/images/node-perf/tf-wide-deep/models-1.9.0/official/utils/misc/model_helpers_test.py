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
""" Tests for Model Helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.misc import model_helpers


class PastStopThresholdTest(tf.test.TestCase):
  """Tests for past_stop_threshold."""

  def test_past_stop_threshold(self):
    """Tests for normal operating conditions."""
    self.assertTrue(model_helpers.past_stop_threshold(0.54, 1))
    self.assertTrue(model_helpers.past_stop_threshold(54, 100))
    self.assertFalse(model_helpers.past_stop_threshold(0.54, 0.1))
    self.assertFalse(model_helpers.past_stop_threshold(-0.54, -1.5))
    self.assertTrue(model_helpers.past_stop_threshold(-0.54, 0))
    self.assertTrue(model_helpers.past_stop_threshold(0, 0))
    self.assertTrue(model_helpers.past_stop_threshold(0.54, 0.54))

  def test_past_stop_threshold_none_false(self):
    """Tests that check None returns false."""
    self.assertFalse(model_helpers.past_stop_threshold(None, -1.5))
    self.assertFalse(model_helpers.past_stop_threshold(None, None))
    self.assertFalse(model_helpers.past_stop_threshold(None, 1.5))
    # Zero should be okay, though.
    self.assertTrue(model_helpers.past_stop_threshold(0, 1.5))

  def test_past_stop_threshold_not_number(self):
    """Tests for error conditions."""
    with self.assertRaises(ValueError):
      model_helpers.past_stop_threshold("str", 1)

    with self.assertRaises(ValueError):
      model_helpers.past_stop_threshold("str", tf.constant(5))

    with self.assertRaises(ValueError):
      model_helpers.past_stop_threshold("str", "another")

    with self.assertRaises(ValueError):
      model_helpers.past_stop_threshold(0, None)

    with self.assertRaises(ValueError):
      model_helpers.past_stop_threshold(0.7, "str")

    with self.assertRaises(ValueError):
      model_helpers.past_stop_threshold(tf.constant(4), None)


if __name__ == "__main__":
  tf.test.main()
