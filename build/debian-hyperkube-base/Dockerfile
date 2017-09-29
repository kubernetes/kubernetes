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

# The samba-common, cifs-utils, and nfs-common packages depend on
# ucf, which itself depends on /bin/bash existing.
# It doesn't seem to actually need bash, however.
RUN ln -s /bin/sh /bin/bash

RUN echo CACHEBUST>/dev/null && clean-install \
    iptables \
    e2fsprogs \
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
    ceph-common

COPY cni-bin/bin /opt/cni/bin
