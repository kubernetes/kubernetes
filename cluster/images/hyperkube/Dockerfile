FROM debian:jessie

RUN apt-get update
RUN apt-get -yy -q install iptables ca-certificates file util-linux

RUN cp /usr/bin/nsenter /nsenter

COPY hyperkube /hyperkube
RUN chmod a+rx /hyperkube

COPY master-multi.json /etc/kubernetes/manifests-multi/master.json
COPY master.json /etc/kubernetes/manifests/master.json

COPY safe_format_and_mount /usr/share/google/safe_format_and_mount
RUN chmod a+rx /usr/share/google/safe_format_and_mount
