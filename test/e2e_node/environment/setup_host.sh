#!/bin/bash

# Copyright 2016 The Kubernetes Authors All rights reserved.
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

# Script used to configure node e2e test hosts from gce base images.
# DISCLAIMER: This script is not actively tested or maintained.  No guarantees that this will work
# on any host environment.  Contributions encouraged!  Send PRs to pwittrock (github.com).
#
# At some point has successfully configured the following distros:
# - ubuntu trusty
# - containervm (no-op)
# - rhel 7
# - centos 7
# - debian jessie

set -e
set -x

# Fixup sudoers require tty
sudo grep -q "# Defaults    requiretty" /etc/sudoers
if [ $? -ne 0 ] ; then
  sudo sed -i 's/Defaults    requiretty/# Defaults    requiretty/' /etc/sudoers
fi

# Install etcd
hash etcd 2>/dev/null
if [ $? -ne 0 ]; then
  curl -L  https://github.com/coreos/etcd/releases/download/v2.2.5/etcd-v2.2.5-linux-amd64.tar.gz -o etcd-v2.2.5-linux-amd64.tar.gz
  tar xzvf etcd-v2.2.5-linux-amd64.tar.gz
  sudo mv etcd-v2.2.5-linux-amd64/etcd* /usr/local/bin/
  sudo chown root:root /usr/local/bin/etcd*
  rm -r etcd-v2.2.5-linux-amd64*
fi

# Install docker
hash docker 2>/dev/null
if [ $? -ne 0 ]; then
  curl -fsSL https://get.docker.com/ | sh
  sudo service docker start
  sudo systemctl enable docker.service
fi

# install lxc
cat /etc/*-release | grep "ID=debian"
if [ $? -ne 0 ]; then
  sudo apt-get install lxc -y
  lxc-checkconfig
  sudo sed -i 's/GRUB_CMDLINE_LINUX="\(.*\)"/GRUB_CMDLINE_LINUX="\1 cgroup_enable=memory"/' /etc/default/grub
  sudo update-grub
fi
