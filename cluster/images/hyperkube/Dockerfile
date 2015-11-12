FROM debian:jessie

RUN DEBIAN_FRONTEND=noninteractive apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get -yy -q \
    install \
    iptables \
    ca-certificates \
    file \
    util-linux \
    socat \
    && DEBIAN_FRONTEND=noninteractive apt-get autoremove -y \
    && DEBIAN_FRONTEND=noninteractive apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN cp /usr/bin/nsenter /nsenter

COPY hyperkube /hyperkube
RUN chmod a+rx /hyperkube

COPY master-multi.json /etc/kubernetes/manifests-multi/master.json
COPY master.json /etc/kubernetes/manifests/master.json

COPY safe_format_and_mount /usr/share/google/safe_format_and_mount
RUN chmod a+rx /usr/share/google/safe_format_and_mount
