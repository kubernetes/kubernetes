# Copyright 2017 The Kubernetes Authors.
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

FROM BASEIMAGE

# TODO(#69896): deprecate the shortened aliases in /
RUN ln -s /hyperkube /apiserver \
 && ln -s /hyperkube /cloud-controller-manager \
 && ln -s /hyperkube /controller-manager \
 && ln -s /hyperkube /kubectl \
 && ln -s /hyperkube /kubelet \
 && ln -s /hyperkube /proxy \
 && ln -s /hyperkube /scheduler \
 && ln -s /hyperkube /usr/local/bin/cloud-controller-manager \
 && ln -s /hyperkube /usr/local/bin/kube-apiserver \
 && ln -s /hyperkube /usr/local/bin/kube-controller-manager \
 && ln -s /hyperkube /usr/local/bin/kube-proxy \
 && ln -s /hyperkube /usr/local/bin/kube-scheduler \
 && ln -s /hyperkube /usr/local/bin/kubectl \
 && ln -s /hyperkube /usr/local/bin/kubelet

RUN echo CACHEBUST>/dev/null && clean-install \
    bash

# The samba-common, cifs-utils, and nfs-common packages depend on
# ucf, which itself depends on /bin/bash.
RUN echo "dash dash/sh boolean false" | debconf-set-selections
RUN DEBIAN_FRONTEND=noninteractive dpkg-reconfigure dash

RUN echo CACHEBUST>/dev/null && clean-install \
    ca-certificates \
    ceph-common \
    cifs-utils \
    conntrack \
    e2fsprogs \
    xfsprogs \
    ebtables \
    ethtool \
    git \
    glusterfs-client \
    iptables \
    ipset \
    jq \
    kmod \
    openssh-client \
    netbase \
    nfs-common \
    socat \
    udev \
    util-linux

COPY cni-bin/bin /opt/cni/bin
