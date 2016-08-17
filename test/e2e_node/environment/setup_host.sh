#!/bin/bash

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

# RHEL os detection
cat /etc/*-release | grep "ID=\"rhel\""
OS_RHEL=$?

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
if $(sudo systemctl status update-engine &>/dev/null); then
  sudo systemctl mask update-engine locksmithd
fi

# Fixup sudoers require tty
sudo grep -q "# Defaults    requiretty" /etc/sudoers
if [ $? -ne 0 ] ; then
  sudo sed -i 's/Defaults    requiretty/# Defaults    requiretty/' /etc/sudoers
fi

# Install etcd
hash etcd 2>/dev/null
if [ $? -ne 0 ]; then
  curl -L  https://github.com/coreos/etcd/releases/download/v3.0.4/etcd-v3.0.4-linux-amd64.tar.gz -o etcd-v3.0.4-linux-amd64.tar.gz
  tar xzvf etcd-v3.0.4-linux-amd64.tar.gz
  sudo mv etcd-v3.0.4-linux-amd64/etcd* /usr/local/bin/
  sudo chown root:root /usr/local/bin/etcd*
  rm -r etcd-v3.0.4-linux-amd64*
fi

# Install nsenter for ubuntu images
cat /etc/*-release | grep "ID=ubuntu"
if [ $? -eq 0 ]; then
  if ! which nsenter > /dev/null; then
     echo "Do not find nsenter. Install it."
     mkdir -p /tmp/nsenter-install
     cd /tmp/nsenter-install
     curl https://www.kernel.org/pub/linux/utils/util-linux/v2.24/util-linux-2.24.tar.gz | tar -zxf-
     sudo apt-get update
     sudo apt-get --yes install make
     sudo apt-get --yes install gcc
     cd util-linux-2.24
     ./configure --without-ncurses
     make nsenter
     sudo cp nsenter /usr/local/bin
     rm -rf /tmp/nsenter-install
   fi
fi

# Install docker
hash docker 2>/dev/null
if [ $? -ne 0 ]; then
  # RHEL platforms should always install from RHEL repository
  # This will install the latest supported stable docker platform on RHEL
  if [ $OS_RHEL -eq 0 ]; then
    sudo yum install -y docker-latest
    sudo groupadd docker
    sudo systemctl enable docker-latest.service
    sudo systemctl start docker-latest.service
  else
    curl -fsSL https://get.docker.com/ | sh
    sudo service docker start
    sudo systemctl enable docker.service
  fi
fi

# Allow jenkins access to docker
sudo usermod -a -G docker jenkins

# install lxc
cat /etc/*-release | grep "ID=debian"
if [ $? -ne 0 ]; then
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
