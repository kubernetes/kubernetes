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

FROM gcr.io/google-containers/debian-base-amd64:0.3
MAINTAINER qlee@google.com

ENV container docker

ARG DEBIAN_FRONTEND=noninteractive
RUN clean-install apt-utils \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg2 \
    software-properties-common \
    curl \
    lxc \
    iptables \
    iproute2 \
    socat \
    util-linux \
    mount ebtables \
    ethtool \
    vim \
    bash \
    procps \
    less \
    kmod \
    systemd \
    systemd-sysv \
    jq

# Install docker.
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && apt-key fingerprint 0EBFCD88 && add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/debian \
    jessie \
    stable"
RUN clean-install docker-ce

# Install the magic wrapper.
COPY dind /bin/dind
COPY dockerd-entrypoint.sh /bin/dockerd-entrypoint.sh

# Remove unwanted systemd services.
RUN for i in /lib/systemd/system/sysinit.target.wants/*; do [ "${i##*/}" = "systemd-tmpfiles-setup.service" ] || rm -f "$i"; done; \
  rm -f /lib/systemd/system/multi-user.target.wants/*;\
  rm -f /etc/systemd/system/*.wants/*;\
  rm -f /lib/systemd/system/local-fs.target.wants/*; \
  rm -f /lib/systemd/system/sockets.target.wants/*udev*; \
  rm -f /lib/systemd/system/sockets.target.wants/*initctl*; \
  rm -f /lib/systemd/system/basic.target.wants/*;\
  rm -f /lib/systemd/system/anaconda.target.wants/*;

# We don't want systemd to start docker for us.
RUN find / -name docker.conf -exec rm {} \;
RUN find / -name docker.service -exec rm {} \;
RUN find / -name docker.socket -exec rm {} \;

# TODO(Q-Lee): SIGRTMIN+3 is the shutdown signal for systemd. Investigate why "STOPSIGNAL SIGRTMIN+3" breaks.

# Define additional metadata for our image.
VOLUME /var/lib/docker
VOLUME /var/lib/kubelet

ENTRYPOINT ["/sbin/init"]
