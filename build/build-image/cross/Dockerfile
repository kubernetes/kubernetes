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

# This file creates a standard build environment for building cross
# platform go binary for the architecture kubernetes cares about.

FROM golang:1.8.3

ENV GOARM 7
ENV KUBE_DYNAMIC_CROSSPLATFORMS \
  armhf \
  arm64 \
  s390x \
  ppc64el

ENV KUBE_CROSSPLATFORMS \
  linux/386 \
  linux/arm linux/arm64 \
  linux/ppc64le \
  linux/s390x \
  darwin/amd64 darwin/386 \
  windows/amd64 windows/386

# Pre-compile the standard go library when cross-compiling. This is much easier now when we have go1.5+
RUN for platform in ${KUBE_CROSSPLATFORMS}; do GOOS=${platform%/*} GOARCH=${platform##*/} go install std; done

# Install g++, then download and install protoc for generating protobuf output
RUN apt-get update \
  && apt-get install -y g++ rsync apt-utils file patch \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/local/src/protobuf \
  && cd /usr/local/src/protobuf \
  && curl -sSL https://github.com/google/protobuf/releases/download/v3.0.0-beta-2/protobuf-cpp-3.0.0-beta-2.tar.gz | tar -xzv \
  && cd protobuf-3.0.0-beta-2 \
  && ./configure \
  && make install \
  && ldconfig \
  && cd .. \
  && rm -rf protobuf-3.0.0-beta-2 \
  && protoc --version

# Use dynamic cgo linking for architectures other than amd64 for the server platforms
# To install crossbuild essential for other architectures add the following repository.
RUN echo "deb http://archive.ubuntu.com/ubuntu xenial main universe" > /etc/apt/sources.list.d/cgocrosscompiling.list \
  && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 40976EAF437D05B5 3B4FE6ACC0B21F32 \
  && apt-get update \
  && apt-get install -y build-essential \
  && for platform in ${KUBE_DYNAMIC_CROSSPLATFORMS}; do apt-get install -y crossbuild-essential-${platform}; done \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# work around 64MB tmpfs size in Docker 1.6
ENV TMPDIR /tmp.k8s

# Get the code coverage tool, goimports, and godep
RUN mkdir $TMPDIR \
  && chmod a+rwx $TMPDIR \
  && chmod o+t $TMPDIR \
  && go get golang.org/x/tools/cmd/cover \
            golang.org/x/tools/cmd/goimports \
            github.com/tools/godep

# Download and symlink etcd. We need this for our integration tests.
RUN export ETCD_VERSION=v3.0.17; \
  mkdir -p /usr/local/src/etcd \
  && cd /usr/local/src/etcd \
  && curl -fsSL https://github.com/coreos/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-linux-amd64.tar.gz | tar -xz \
  && ln -s ../src/etcd/etcd-${ETCD_VERSION}-linux-amd64/etcd /usr/local/bin/
