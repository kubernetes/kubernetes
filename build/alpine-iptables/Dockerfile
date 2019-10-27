# Copyright 2019 The Kubernetes Authors.
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

RUN apk add -U --no-cache \
    ebtables \
    iptables \
    ip6tables \
    conntrack-tools \
    ipset \
    kmod \
    iproute2

# Install iptables wrapper scripts to detect the correct iptables mode
# the first time any of them is run
# Alpine PATH is PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# and iptables binaries are installed in /sbin by default
COPY iptables-wrapper /usr/sbin/iptables-wrapper

RUN ln -s /usr/sbin/iptables-wrapper /usr/sbin/iptables \
    && ln -s /usr/sbin/iptables-wrapper /usr/sbin/ip6tables \
    && ln -s /usr/sbin/iptables-wrapper /usr/sbin/iptables-restore \
    && ln -s /usr/sbin/iptables-wrapper /usr/sbin/ip6tables-restore \
    && ln -s /usr/sbin/iptables-wrapper /usr/sbin/iptables-save \
    && ln -s /usr/sbin/iptables-wrapper /usr/sbin/ip6tables-save
