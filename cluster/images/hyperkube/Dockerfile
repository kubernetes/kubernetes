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

RUN DEBIAN_FRONTEND=noninteractive apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get -yy -q \
    install \
    iptables \
    ethtool \
    ca-certificates \
    file \
    util-linux \
    socat \
    curl \
    && DEBIAN_FRONTEND=noninteractive apt-get autoremove -y \
    && DEBIAN_FRONTEND=noninteractive apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN cp /usr/bin/nsenter /nsenter

COPY hyperkube /hyperkube
RUN chmod a+rx /hyperkube

COPY master-multi.json /etc/kubernetes/manifests-multi/master.json
COPY kube-proxy.json /etc/kubernetes/manifests-multi/kube-proxy.json

COPY master.json /etc/kubernetes/manifests/master.json
COPY etcd.json /etc/kubernetes/manifests/etcd.json
COPY kube-proxy.json /etc/kubernetes/manifests/kube-proxy.json

COPY safe_format_and_mount /usr/share/google/safe_format_and_mount
RUN chmod a+rx /usr/share/google/safe_format_and_mount

COPY setup-files.sh /setup-files.sh
RUN chmod a+rx /setup-files.sh

COPY make-ca-cert.sh /make-ca-cert.sh
RUN chmod a+x /make-ca-cert.sh
