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

FROM BASEIMAGE

# If we're building for another architecture than amd64, the CROSS_BUILD_ placeholder is removed so e.g. CROSS_BUILD_COPY turns into COPY
# If we're building normally, for amd64, CROSS_BUILD lines are removed
CROSS_BUILD_COPY qemu-ARCH-static /usr/bin/

RUN DEBIAN_FRONTEND=noninteractive apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get -yy -q install \
    iptables \
    ethtool \
    ca-certificates \
    file \
    util-linux \
    socat \
    curl \
    git \
    && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get autoremove -y \
    && DEBIAN_FRONTEND=noninteractive apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN cp /usr/bin/nsenter /nsenter

# Copy the hyperkube binary
COPY hyperkube /hyperkube

# Manifests for the docker guide
COPY master.json /etc/kubernetes/manifests/master.json
COPY etcd.json /etc/kubernetes/manifests/etcd.json
COPY kube-proxy.json /etc/kubernetes/manifests/kube-proxy.json

# Manifests for the docker-multinode guide
COPY master-multi.json /etc/kubernetes/manifests-multi/master.json
COPY kube-proxy.json /etc/kubernetes/manifests-multi/kube-proxy.json

# Other required scripts for the setup
COPY safe_format_and_mount /usr/share/google/safe_format_and_mount
COPY setup-files.sh /setup-files.sh
COPY make-ca-cert.sh /make-ca-cert.sh

# easy-rsa package required by make-ca-cert
ADD https://storage.googleapis.com/kubernetes-release/easy-rsa/easy-rsa.tar.gz /root/kube/

RUN mkdir -p /opt/cni
RUN curl https://storage.googleapis.com/kubernetes-release/network-plugins/cni-c864f0e1ea73719b8f4582402b0847064f9883b0.tar.gz \
  | tar xzv -C /opt/cni

# Make scripts executable
RUN chmod a+rx \
        /hyperkube \
        /usr/share/google/safe_format_and_mount \
        /setup-files.sh \
        /make-ca-cert.sh
