#!/usr/bin/env python3

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

import argparse
import os
import sys
import util
import yaml


def get_kernel_version():
  os_release = os.uname()[2]
  fields = os_release.split('.')
  if len(fields) < 2:
    sys.exit('Failed to parse OS release %s' % os_release)
  return fields[0], fields[1]


def get_namespaced_sysctl_names(namespaced_sysctl_names):
  names = set()
  kernel_major, kernel_minor = get_kernel_version()
  for entry in yaml.load(open(namespaced_sysctl_names, 'r')):
    fields = entry['kernel'].split('.')
    if len(fields) < 2:
      sys.exit('Found invalid kernel %s in %s' %
               (entry['kernel'], namespaced_sysctl_names))
    major, minor = fields[0], fields[1]
    if kernel_major < major:
      continue
    if kernel_major == major and kernel_minor < minor:
      continue
    names = set(entry['namespaced'])
    break
  else:
    sys.exit('Failed to find namespaced sysctls for kernel %s.%s in %s' %
             (kernel_major, kernel_minor, namespaced_sysctl_names))
  return names


def print_sysctls(sysctls):
  result = []
  for k in sorted(sysctls):
    result.append('%s=%s' % (k, sysctls[k]))
  print(','.join(result))


def main(sysctl_defaults, sysctl_overrides, pod_sysctl_overrides, namespaced_sysctl_names):
  # Load the sysctl defaults.
  defaults = util.get_sysctls_from_file(sysctl_defaults)
  if not defaults:
    defaults = {}

  # Parse the pod sysctl overrides.
  pod_overrides = util.parse_sysctl_overrides(pod_sysctl_overrides)

  # Apply the pod default overrides on top of the defaults.
  for k, v in pod_overrides.items():
    defaults[k] = v

  # Parse the sysctl overrides.
  user_overrides = util.parse_sysctl_overrides(sysctl_overrides)

  # Apply the user overrides on top of the defaults and pod overrides.
  for k, v in user_overrides.items():
    defaults[k] = v

  # Load the namespaced sysctl names.
  names = get_namespaced_sysctl_names(namespaced_sysctl_names)

  # Extract the namespaced sysctls.
  namespaced_sysctls = {}
  for k, v in defaults.items():
    if k in names:
      namespaced_sysctls[k] = v
  print_sysctls(namespaced_sysctls)


if __name__ == '__main__':
  PARSER = argparse.ArgumentParser(
      description='Extract the namespaced kernel parameters from the values '
      'specified in the YAML file --sysctl-defaults and the '
      'content specified via --sysctl-overrides and --pod-sysctl-overrides, '
      'according to the list in the YAML file --namespaced-sysctl-names.')
  PARSER.add_argument(
      '--sysctl-defaults',
      type=str,
      required=True,
      help='Path to the YAML file containing the default value of the '
      'kernel parameters.')
  PARSER.add_argument(
      '--sysctl-overrides',
      type=str,
      required=True,
      help='List of kernel parameter overrides that will be layered on top '
      'of the default values and pod-sysctl-overrides. Must be specified as '
      'key=value pairs separated by ","')
  PARSER.add_argument(
      '--pod-sysctl-overrides',
      type=str,
      required=False,
      help='List of kernel parameter overrides that will be layered on top '
      'of the default values. This may be overridden by what is given in '
      '--sysctl-overrides. Must be specified as key=value pairs separated '
      'by ","')
  PARSER.add_argument(
      '--namespaced-sysctl-names',
      type=str,
      required=True,
      help='Path to the YAML file containing the list of namespaced kernel '
      'parameter names.')
  ARGS = PARSER.parse_args()

  main(ARGS.sysctl_defaults, ARGS.sysctl_overrides, ARGS.pod_sysctl_overrides,
       ARGS.namespaced_sysctl_names)
