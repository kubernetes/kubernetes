#!/usr/bin/env python

# Copyright 2021 The Kubernetes Authors.
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
import logging
import sys
import yaml

logging.basicConfig(level=logging.DEBUG)


def parse_yaml_file(file_path):
  with open(file_path, 'r') as f:
    sysctls = yaml.safe_load(f)
    return sysctls


def merge_sysctls(new_namespaced_sysctls_file_path,
                  existing_namespaced_sysctls_file_path):
  new_namespaced_sysctls = parse_yaml_file(new_namespaced_sysctls_file_path)
  new_namespaced_sysctls_kernel = new_namespaced_sysctls['kernel']
  logging.info('New namespaced sysctls are on kernel version %s',
               new_namespaced_sysctls_kernel)

  existing_namespaced_sysctls = parse_yaml_file(
      existing_namespaced_sysctls_file_path)

  sorted_by_kernel_version = sorted(
      existing_namespaced_sysctls, key=lambda k: k['kernel'], reverse=True)

  existing_newest_sysctls = sorted_by_kernel_version[0]
  existing_newest_sysctl_set_kernel = existing_newest_sysctls['kernel']

  logging.info(
      'Diffing new sysctls with kernel version: %s with existing sysctls on kernel version: %s',
      new_namespaced_sysctls_kernel, existing_newest_sysctl_set_kernel)

  existing_set = set(existing_newest_sysctls['namespaced'])
  new_set = set(new_namespaced_sysctls['namespaced'])

  added_namespaced_sysctls = new_set - (existing_set & new_set)
  removed_sysctls = existing_set - (existing_set & new_set)

  print('Added:')
  print('\n'.join(['\t' + s for s in added_namespaced_sysctls]))
  print('Removed:')
  print('\n'.join(['\t' + s for s in removed_sysctls]))

  msg = 'Does the diff look reasonable? Continue with the merge?'
  msg += ('\nPlease remember to include the diff below in a CL updating the '
          'namespaced sysctls')
  continue_with_merge = input('%s (y/N) ' % msg).lower() == 'y'

  if not continue_with_merge:
    logging.info('exiting...')
    sys.exit(1)

  merged_sysctls = []
  merged_sysctls.extend(existing_namespaced_sysctls)
  merged_sysctls.append(new_namespaced_sysctls)
  merged_sysctls = sorted(
      merged_sysctls, key=lambda k: k['kernel'], reverse=True)

  # we want to keep comments in the file (at the top file), so we filter and include them
  comment_lines = []
  with open(existing_namespaced_sysctls_file_path) as f:
    comment_lines = [
        line.strip() for line in f.readlines() if line.startswith('#')
    ]

  merged_dump = yaml.dump(merged_sysctls, default_flow_style=False)

  with open(existing_namespaced_sysctls_file_path, 'w') as f:
    f.write('\n'.join(comment_lines))
    f.write('\n')
    f.write(merged_dump)

  logging.info('Done! Updated %s with the new merged sysctls',
               existing_namespaced_sysctls_file_path)


def main():
  parser = argparse.ArgumentParser(
      description='Merge new namespaced sysctls with existing namespaced sysctls file'
  )
  parser.add_argument(
      '--new-namespaced-sysctls',
      type=str,
      required=True,
      help='Path to new namespaced sysctls file')
  parser.add_argument(
      '--existing-namespaced-sysctls',
      type=str,
      required=True,
      help='Path to existing namespaced sysctls file')
  args = parser.parse_args()

  merge_sysctls(args.new_namespaced_sysctls, args.existing_namespaced_sysctls)


if __name__ == '__main__':
  main()
