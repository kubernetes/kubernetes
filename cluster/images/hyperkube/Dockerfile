FROM google/debian:wheezy

RUN apt-get update
RUN apt-get -yy -q install iptables
COPY hyperkube /hyperkube
RUN chmod a+rx /hyperkube

COPY master.json /etc/kubernetes/manifests/master.json

