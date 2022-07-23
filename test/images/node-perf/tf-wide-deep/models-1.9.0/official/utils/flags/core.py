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
"""Public interface for flag definition.

See _example.py for detailed instructions on defining flags.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import sys

from absl import app as absl_app
from absl import flags

from official.utils.flags import _base
from official.utils.flags import _benchmark
from official.utils.flags import _conventions
from official.utils.flags import _misc
from official.utils.flags import _performance


def set_defaults(**kwargs):
  for key, value in kwargs.items():
    flags.FLAGS.set_default(name=key, value=value)


def parse_flags(argv=None):
  """Reset flags and reparse. Currently only used in testing."""
  flags.FLAGS.unparse_flags()
  absl_app.parse_flags_with_usage(argv or sys.argv)


def register_key_flags_in_core(f):
  """Defines a function in core.py, and registers its key flags.

  absl uses the location of a flags.declare_key_flag() to determine the context
  in which a flag is key. By making all declares in core, this allows model
  main functions to call flags.adopt_module_key_flags() on core and correctly
  chain key flags.

  Args:
    f:  The function to be wrapped

  Returns:
    The "core-defined" version of the input function.
  """

  def core_fn(*args, **kwargs):
    key_flags = f(*args, **kwargs)
    [flags.declare_key_flag(fl) for fl in key_flags]  # pylint: disable=expression-not-assigned
  return core_fn


define_base = register_key_flags_in_core(_base.define_base)
# Remove options not relevant for Eager from define_base().
define_base_eager = register_key_flags_in_core(functools.partial(
    _base.define_base, epochs_between_evals=False, stop_threshold=False,
    multi_gpu=False, hooks=False))
define_benchmark = register_key_flags_in_core(_benchmark.define_benchmark)
define_image = register_key_flags_in_core(_misc.define_image)
define_performance = register_key_flags_in_core(_performance.define_performance)


help_wrap = _conventions.help_wrap


get_num_gpus = _base.get_num_gpus
get_tf_dtype = _performance.get_tf_dtype
get_loss_scale = _performance.get_loss_scale
DTYPE_MAP = _performance.DTYPE_MAP
