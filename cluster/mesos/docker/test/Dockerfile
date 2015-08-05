FROM golang:1.4.2
MAINTAINER Mesosphere <support@mesosphere.io>

# docker.io is suppossed to be in backports, but it's not there yet.
# https://github.com/docker/docker/issues/13253
# http://docs.docker.com/installation/debian/#debian-jessie-80-64-bit
#RUN echo "deb http://httpredir.debian.org/debian jessie-backports main" >> /etc/apt/sources.list
#RUN echo "deb http://http.debian.net/debian jessie-backports main" >> /etc/apt/sources.list

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -qqy \
        wget \
        curl \
        g++ \
        make \
        mercurial \
        git \
        rsync \
        patch \
        python \
        python-pip \
        apt-transport-https \
        && \
    apt-get clean

# Install latest Docker
# RUN curl -sSL https://get.docker.com/ubuntu/ | sh

# Install Docker 1.6.2 (docker 1.7 has regressions)
RUN echo deb https://get.docker.com/ubuntu docker main > /etc/apt/sources.list.d/docker.list && \
    apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 36A1D7869245C8950F966E92D8576A8BA88D21E9 && \
    apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -qqy \
        lxc-docker-1.6.2 \
        && \
    apt-get clean

RUN pip install -U docker-compose

RUN go get github.com/tools/godep

RUN mkdir -p /go/src/github.com/GoogleCloudPlatform/kubernetes
WORKDIR /go/src/github.com/GoogleCloudPlatform/kubernetes

COPY ./bin/* /usr/local/bin/

RUN install-etcd.sh

ENTRYPOINT [ "bash" ]
