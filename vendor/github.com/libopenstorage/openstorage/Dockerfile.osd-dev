FROM golang:1.6.2
MAINTAINER gou@portworx.com

EXPOSE 9005
RUN \
  apt-get update -yq && \
  apt-get install -yq --no-install-recommends \
    btrfs-tools \
    ca-certificates && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN \
  curl -sSL https://get.docker.com/builds/Linux/x86_64/docker-1.10.3 > /bin/docker && \
  chmod +x /bin/docker
RUN mkdir -p /go/src/github.com/libopenstorage/openstorage
ADD . /go/src/github.com/libopenstorage/openstorage/
WORKDIR /go/src/github.com/libopenstorage/openstorage
