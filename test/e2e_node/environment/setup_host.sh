#!/usr/bin/env bash

# Copyright 2016 The Kubernetes Authors.
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

# On a systemd environment, enable cpu and memory accounting for all processes by default.
if [ -d /etc/systemd ]; then
  cat <<EOF >kubernetes-accounting.conf
[Manager]
DefaultCPUAccounting=yes
DefaultMemoryAccounting=yes
EOF
  sudo mkdir -p /etc/systemd/system.conf.d/
  sudo cp kubernetes-accounting.conf /etc/systemd/system.conf.d
  sudo systemctl daemon-reload
fi

# For coreos, disable updates
if sudo systemctl status update-engine &>/dev/null; then
  sudo systemctl mask update-engine locksmithd
fi

# Fixup sudoers require tty
if ! sudo grep -q "# Defaults    requiretty" /etc/sudoers; then
  sudo sed -i 's/Defaults    requiretty/# Defaults    requiretty/' /etc/sudoers
fi

# Install nsenter for ubuntu images
if cat /etc/*-release | grep "ID=ubuntu"; then
  if ! which nsenter > /dev/null; then
     echo "Do not find nsenter. Install it."
     NSENTER_BUILD_DIR=$(mktemp -d /tmp/nsenter-build-XXXXXX)
     cd "$NSENTER_BUILD_DIR" || exit 1
     curl https://www.kernel.org/pub/linux/utils/util-linux/v2.31/util-linux-2.31.tar.gz | tar -zxf-
     sudo apt-get update
     sudo apt-get --yes install make
     sudo apt-get --yes install gcc
     cd util-linux-2.31 || exit 1
     ./configure --without-ncurses
     make nsenter
     sudo cp nsenter /usr/local/bin
     rm -rf "$NSENTER_BUILD_DIR"
   fi
fi

# Install docker
if ! hash containerd 2>/dev/null; then
  echo "Please install containerd, see the getting started guide here: https://github.com/containerd/containerd/blob/main/docs/getting-started.md"
  echo "For a docker CLI replacement, we suggest nerdctl: https://github.com/containerd/nerdctl#install"
fi

# Allow jenkins access to docker
id jenkins || sudo useradd jenkins -m
sudo usermod -a -G docker jenkins

# install lxc
if ! cat /etc/*-release | grep "ID=debian"; then
  hash apt-get 2>/dev/null
  if [ $? -ne 1 ]; then
    sudo apt-get install lxc -y
    lxc-checkconfig
    sudo sed -i 's/GRUB_CMDLINE_LINUX="\(.*\)"/GRUB_CMDLINE_LINUX="\1 cgroup_enable=memory"/' /etc/default/grub
    sudo update-grub
  fi
fi

# delete init kubelet from containervm so that is doesn't startup
if [ -f /etc/init.d/kubelet ]; then
  sudo rm /etc/init.d/kubelet
fi
