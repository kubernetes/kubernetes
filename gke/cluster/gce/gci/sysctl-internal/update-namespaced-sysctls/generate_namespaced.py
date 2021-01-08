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
import os
import subprocess
import sys
import yaml
import logging

logging.basicConfig(level=logging.DEBUG)

# The list of sysctls that shouldn't be treated as namespaced parameters.
DENYLIST = set([
    'net.ipv4.ipfrag_high_thresh',  # flaky: sometimes failed to set.
    'net.ipv6.ip6frag_high_thresh',  # flaky: sometimes failed to set.
    'net.netfilter.nf_conntrack_count',  # ignore status value.
    'net.netfilter.nf_conntrack_frag6_high_thresh',  # flaky: sometimes failed to set.
    'net.ipv6.conf.all.stable_secret',
    'net.ipv6.conf.default.stable_secret',
    'net.ipv6.conf.eth0.stable_secret',
    'net.ipv6.conf.lo.stable_secret',
    'net.ipv6.icmp.ratemask',  # unrecognized format
])

# The list of prefixes of the sysctls that shouldn't be treated as namespaced
# parameters.
DENYLIST_PREFIX = set([
    'net.ipv4.conf.eth0.',
    'net.ipv4.conf.lo.',
    'net.ipv4.neigh.eth0.',
    'net.ipv4.neigh.lo.',
    'net.ipv6.conf.eth0.',
    'net.ipv6.conf.lo.',
    'net.ipv6.neigh.eth0.',
    'net.ipv6.neigh.lo.',
])

SPECIAL_SYSCTLS = {
    'net.ipv4.tcp_congestion_control': ('htcp', 'cubic'),
    'net.ipv4.tcp_fastopen_key': ('00000000-00000000-00000000-00000000',
                                  '00000000-00000000-00000000-00000001'),
    'net.ipv4.tcp_min_snd_mss':
        ('48', '96'),  # newly added when fixing the TCP SACK security issue.
    'net.ipv4.ip_local_reserved_ports': ('36741', '36742'),
    'net.ipv4.ipfrag_high_thresh': ('8388608', '262144'),
    'net.ipv4.vs.sync_ports': ('1', '2'),
    'net.ipv6.conf.all.mtu': ('1500', '1280'),
    'net.ipv6.conf.default.mtu': ('1500', '1280'),
    'net.ipv6.conf.eth0.mtu': ('1500', '1280'),
    'net.ipv6.ip6frag_high_thresh': ('8388608', '262144'),
    'net.ipv6.neigh.eth0.locktime': ('0', '1000'),
    'net.ipv6.neigh.lo.locktime': ('0', '1000'),
    'net.netfilter.nf_log.0': ('NONE', 'nfnetlink_log'),
    'net.netfilter.nf_log.1': ('NONE', 'nfnetlink_log'),
    'net.netfilter.nf_log.2': ('NONE', 'nfnetlink_log'),
    'net.netfilter.nf_log.3': ('NONE', 'nfnetlink_log'),
    'net.netfilter.nf_log.4': ('NONE', 'nfnetlink_log'),
    'net.netfilter.nf_log.5': ('NONE', 'nfnetlink_log'),
    'net.netfilter.nf_log.6': ('NONE', 'nfnetlink_log'),
    'net.netfilter.nf_log.7': ('NONE', 'nfnetlink_log'),
    'net.netfilter.nf_log.8': ('NONE', 'nfnetlink_log'),
    'net.netfilter.nf_log.9': ('NONE', 'nfnetlink_log'),
    'net.netfilter.nf_log.10': ('NONE', 'nfnetlink_log'),
    'net.netfilter.nf_log.11': ('NONE', 'nfnetlink_log'),
    'net.netfilter.nf_log.12': ('NONE', 'nfnetlink_log'),
}


class IntValue(object):

  def __init__(self, key, value):
    self.key = key
    self.value = value

  def generate_new_value(self):
    return str(self.value // 2)


class BoolValue(object):

  def __init__(self, key, value):
    self.key = key
    self.value = value

  def generate_new_value(self):
    if self.value == -1 or self.value == 1:
      return '0'
    else:
      return '1'


class SpecialValue(object):

  def __init__(self, key, value):
    self.key = key
    self.value = value

  @staticmethod
  def is_special(key):
    return key in SPECIAL_SYSCTLS

  def generate_new_value(self):
    values = SPECIAL_SYSCTLS[self.key]
    try:
      i = values.index(self.value)
    except ValueError:
      i = -1
    return values[(i + 1) % 2]


class TupleValue(object):

  def __init__(self, key, value):
    self.key = key
    self.value = []

    # Check whether this is a parameter that needs special handling.
    if SpecialValue.is_special(key):
      self.value.append(SpecialValue(key, value))
      return

    for field in value.split('\t'):
      field = field.strip()
      if field == '0' or field == '1' or field == '-1':
        self.value.append(BoolValue(key, int(field)))
      else:
        self.value.append(IntValue(key, int(field)))

  def generate_new_value(self):
    return ' '.join(list(map(lambda v: v.generate_new_value(), self.value)))


def get_kernel_version():
  os_release = os.uname()[2]
  fields = os_release.split('.')
  if len(fields) < 2:
    sys.exit('Failed to parse OS release %s' % os_release)
  return fields[0] + '.' + fields[1]


def is_in_denylist(sysctl):
  """Return True if the sysctl is excluded from monitoring."""
  if sysctl in DENYLIST:
    return True
  for prefix in DENYLIST_PREFIX:
    if sysctl.startswith(prefix):
      return True
  return False


def key_to_path(key):
  path = key.replace('.', '/')
  return os.path.join('/proc/sys/', path)


def get_sysctl_in_host(key):
  cmd = ['sudo', 'cat', key_to_path(key)]
  return subprocess.check_output(cmd).strip().decode()


def get_sysctls_in_container(container_id):
  cmd = [
      'sudo', 'docker', 'exec', container_id, 'sysctl', '-aNr',
      '^kernel\.shm|^kernel\.msg|^kernel\.sem|^fs\.mqueue\.|^net\.'
  ]
  with open(os.devnull, 'w') as FNULL:
    output = subprocess.check_output(cmd, stderr=FNULL)
    return filter(None, output.decode().split('\n'))


def set_sysctl_in_container(key,
                            value_in_container,
                            container_id,
                            value_in_host=None):
  # Use procfs instead of `sysctl -w` because `sysctl` may unexpectedly return
  # zero value on errors.
  set_cmd = 'echo "%s" > %s' % (value_in_container, key_to_path(key))
  cmd = ['sudo', 'docker', 'exec', container_id, '/bin/bash', '-c', set_cmd]
  try:
    subprocess.check_output(cmd, stderr=subprocess.STDOUT)
  except subprocess.CalledProcessError as e:
    logging.info('failed to set "%s=%s" in container: %s', key,
                 value_in_container, e.output.strip())
    return False

  if not value_in_host:
    return True

  # Verify it's actually changed in the container because the above command may
  # succeed even if the value failed to be changed.
  cmd = ['sudo', 'docker', 'exec', container_id, 'cat', key_to_path(key)]
  output = subprocess.check_output(cmd).decode().strip().replace('\t', ' ')
  if value_in_host == output:
    logging.error(
        'failed to set "%s=%s" in container: current value "%s" remains unchanged',
        key, value_in_container, output)
    return False

  return True


def is_namespaced(key, container_id):
  logging.info('Checking if %s is namespaced', key)
  if is_in_denylist(key):
    return False

  # Get the current value in host namespace.
  old_value_in_host = get_sysctl_in_host(key)

  # Set the parameter to a different value in the container namespace.
  o = TupleValue(key, old_value_in_host)
  value_in_container = o.generate_new_value()

  if not set_sysctl_in_container(
      key, value_in_container, container_id, value_in_host=old_value_in_host):
    # This may or may not be namespaced but no matter which, it can't be set in
    # container namespace.
    return False

  # Check whether the value in host namespace was changed to the value set in
  # the container namespace.
  new_value_in_host = get_sysctl_in_host(key)
  if old_value_in_host == new_value_in_host:
    # The value in host namespace did not change so this is a namespaced
    # parameter.
    return True
  elif new_value_in_host == value_in_container:
    # Restore the sysctl back on the host namespace to the old value.
    set_sysctl_in_container(key, old_value_in_host, container_id)
    cmd = ['sudo', 'sysctl', '-w', '%s=%s' % (key, old_value_in_host)]
    subprocess.check_output(cmd)
    # The change inside the container namespace is visible to host namespace so
    # this is a unnamespaced parameter.
    return False
  else:
    logging.error('unexpected change on %s', key)
    return False


def generate_namespaced_sysctls(out_file):
  output = {}
  output['kernel'] = get_kernel_version()

  logging.info('Starting a Docker container...')
  cmd = ['sudo', 'docker', 'run', '-id', '--privileged', 'ubuntu']
  container_id = subprocess.check_output(cmd).strip()

  logging.info('Getting the initial list of sysctls for checking...')
  sysctls = get_sysctls_in_container(container_id)

  namespaced = [s for s in sysctls if is_namespaced(s, container_id)]
  output['namespaced'] = namespaced

  logging.info('Stopping the docker container...')
  cmd = ['sudo', 'docker', 'stop', container_id]
  subprocess.check_output(cmd)

  with open(out_file, 'w') as f:
    yaml.dump(output, f, default_flow_style=False)

  logging.info('Done, wrote namespaced sysctls to %s', out_file)


def main():
  parser = argparse.ArgumentParser(description='Output namespaced sysctls')
  parser.add_argument(
      '--out-file',
      type=str,
      required=True,
      help='Output file where namespaced sysctls will be written to')
  args = parser.parse_args()

  generate_namespaced_sysctls(args.out_file)


if __name__ == '__main__':
  main()
