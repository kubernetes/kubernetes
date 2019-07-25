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

import argparse
import util


def dump_to_file(conf, filename):
  with open(filename, 'w') as output_file:
    output_file.write('\n'.join(conf))
    output_file.write('\n')


def main(sysctl_defaults, sysctl_overrides, output_defaults, output_overrides):
  # Dump the sysctl defaults to the .conf file.
  output = []
  defaults = util.get_sysctls_from_file(sysctl_defaults)
  if defaults:
    for k in sorted(defaults):
      output.append('%s=%s' % (k, defaults[k]))
    dump_to_file(output, output_defaults)

  # Parse the sysctl overrides and dump them to the .conf file.
  overrides = util.parse_sysctl_overrides(sysctl_overrides)
  if overrides:
    output = []
    for k, v in overrides.iteritems():
      output.append('%s=%s' % (k, v))
    dump_to_file(output, output_overrides)


if __name__ == '__main__':
  PARSER = argparse.ArgumentParser(
      description='Generate the defaults and overrides sysctl conf files '
      'from the values in the YAML file --sysctl-defaults and '
      'the content specified via --sysctl-overrides.')
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
      'of the default values. Must be specified as key=value pairs '
      'separated by ","')
  PARSER.add_argument(
      '--output-defaults',
      type=str,
      required=True,
      help='Path to the output kernel parameter conf file containing the '
      'sysctl default values.')
  PARSER.add_argument(
      '--output-overrides',
      type=str,
      required=True,
      help='Path to the output kernel parameter conf file containing the '
      'sysctl overrides.')
  ARGS = PARSER.parse_args()

  main(ARGS.sysctl_defaults, ARGS.sysctl_overrides, ARGS.output_defaults,
       ARGS.output_overrides)
