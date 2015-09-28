FROM debian:jessie

COPY kubemark.sh /kubemark.sh
COPY kubernetes-server-linux-amd64.tar.gz /tmp/kubemark.tar.gz
COPY build-kubemark.sh /build-kubemark.sh

RUN /build-kubemark.sh
