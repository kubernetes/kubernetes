#!/usr/bin/env python

# Copyright 2019 The Kubernetes Authors.
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


from multiprocessing import cpu_count
import yaml

EVAL_LOCALS_DICT = {
  'cpu_count': cpu_count,
  'max': max,
}

def parse_sysctl_overrides(sysctl_overrides):
  overrides = {}
  parts = sysctl_overrides.split(',')
  for part in parts:
    pair = part.split('=')
    if len(pair) != 2:
      continue
    k, v = pair[0], pair[1]
    overrides[k] = v
  return overrides


def eval_expressions(sysctls):
  """Evaluate the expressions in the sysctl values.

  For example, given
    net.netfilter.nf_conntrack_max: "max(128*1024,32*1024*cpu_count())"
  this function will return
    net.netfilter.nf_conntrack_max: "524288"
  on a machine with 16 CPUs, assuming cpuset is not used.

  Args:
    sysctls: Dict of sysctls with value expressions.

  Returns:
    Sysctls with evaluated values.
  """
  for key, value in sysctls.iteritems():
    # pylint: disable=eval-used
    fields = map(lambda x: str(eval(
      x, {'__builtins__': None}, EVAL_LOCALS_DICT)), value.split(' '))
    sysctls[key] = ' '.join(fields)


def get_sysctls_from_file(filename):
  sysctls = yaml.load(open(filename, 'r'))
  if not sysctls:
    return sysctls
  # Evaluate the expressions in the default sysctl values. This has no security
  # concerns as the default sysctls are hardcoded by GKE.
  eval_expressions(sysctls)
  return sysctls
