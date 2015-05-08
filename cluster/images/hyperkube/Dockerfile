FROM google/debian:wheezy

RUN apt-get update
RUN apt-get -yy -q install iptables ca-certificates
COPY hyperkube /hyperkube
RUN chmod a+rx /hyperkube


COPY master-multi.json /etc/kubernetes/manifests-multi/master.json
COPY master.json /etc/kubernetes/manifests/master.json
