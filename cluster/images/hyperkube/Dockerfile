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

FROM BASEIMAGE

# If we're building for another architecture than amd64, the CROSS_BUILD_ placeholder is removed so e.g. CROSS_BUILD_COPY turns into COPY
# If we're building normally, for amd64, CROSS_BUILD lines are removed
CROSS_BUILD_COPY qemu-ARCH-static /usr/bin/

RUN DEBIAN_FRONTEND=noninteractive apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get -yy -q install \
    iptables \
    ebtables \
    ethtool \
    ca-certificates \
    conntrack \
    util-linux \
    socat \
    git \
    jq \
    nfs-common \
    glusterfs-client \
    cifs-utils \
    && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get autoremove -y \
    && DEBIAN_FRONTEND=noninteractive apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* # CACHEBUST

RUN cp /usr/bin/nsenter /nsenter

# Manifests for the docker guide
COPY static-pods/master.json \
     static-pods/etcd.json \
     static-pods/addon-manager-singlenode.json \
     static-pods/kube-proxy.json \
        /etc/kubernetes/manifests/

# Manifests for the docker-multinode guide
COPY static-pods/master-multi.json \
     static-pods/addon-manager-multinode.json \
        /etc/kubernetes/manifests-multi/

# Copy over all addons
COPY addons /etc/kubernetes/addons

# Copy other required scripts for the setup
COPY setup-files.sh make-ca-cert.sh copy-addons.sh /

# easy-rsa package required by make-ca-cert
ADD https://storage.googleapis.com/kubernetes-release/easy-rsa/easy-rsa.tar.gz /root/kube/

# Copy the the cni-bin folder into /opt/cni/bin
COPY cni-bin/bin /opt/cni/bin

# Copy overlay configuration to default directory
COPY cni-conf /etc/cni/net.d

# Create symlinks for each hyperkube server
# TODO: replace manual symlink creation with --make-symlink command once
# cross-building with qemu supports go binaries. See #28702
# RUN /hyperkube --make-symlinks
RUN ln -s /hyperkube /apiserver \
 && ln -s /hyperkube /controller-manager \
 && ln -s /hyperkube /federation-apiserver \
 && ln -s /hyperkube /federation-controller-manager \
 && ln -s /hyperkube /kubectl \
 && ln -s /hyperkube /kubelet \
 && ln -s /hyperkube /proxy \
 && ln -s /hyperkube /scheduler

# Copy the hyperkube binary
COPY hyperkube /hyperkube

