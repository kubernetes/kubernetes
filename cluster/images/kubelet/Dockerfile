FROM debian:jessie

RUN DEBIAN_FRONTEND=noninteractive apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get -yy -q \
    install \
    iptables \
    ethtool \
    && DEBIAN_FRONTEND=noninteractive apt-get autoremove -y \
    && DEBIAN_FRONTEND=noninteractive apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY kubelet /kubelet
RUN chmod a+rx /kubelet

COPY kubelet-runner.sh /kubelet-runner.sh
RUN chmod a+rx /kubelet-runner.sh

VOLUME /sys
VOLUME /dev
VOLUME /var/lib/docker
VOLUME /var/lib/kubelet
VOLUME /var/run
