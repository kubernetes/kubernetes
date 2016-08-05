# Copyright 2016 The Kubernetes Authors.
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

FROM golang:1.6.0
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

# Install Docker 1.8.1 explicitly
RUN apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D && \
    mkdir -p /etc/apt/sources.list.d && \
    echo deb https://apt.dockerproject.org/repo ubuntu-trusty main > /etc/apt/sources.list.d/docker.list && \
    apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -qqy \
        docker-engine=1.8.1-0~trusty \
        && \
    apt-get clean

RUN pip install -U docker-compose==1.5.0

RUN go get github.com/tools/godep

RUN mkdir -p /go/src/github.com/GoogleCloudPlatform/kubernetes
WORKDIR /go/src/github.com/GoogleCloudPlatform/kubernetes

COPY ./bin/* /usr/local/bin/

RUN install-etcd.sh

ENTRYPOINT [ "bash" ]
