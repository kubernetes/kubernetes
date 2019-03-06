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

ARG ARCH
ARG BASE_IMAGE=golang:1.12.6
FROM ${BASE_IMAGE} as fetch_cover_and_goimports

# Fetch the code coverage tool and goimports
RUN GOOS=linux GOARCH=${ARCH} go get golang.org/x/tools/cmd/cover \
           golang.org/x/tools/cmd/goimports \
    && go clean -cache

FROM --platform=linux/${ARCH} ${BASE_IMAGE}
COPY --from=fetch_cover_and_goimports /go/ /go/

ARG ARCH
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

CROSS_BUILD_COPY qemu-QEMUARCH-static /usr/bin/

RUN go version

# Pre-compile the standard go library when cross-compiling. This is much easier now when we have go1.5+
RUN ([ "$(uname -m)" = "x86_64" ] && for platform in ${KUBE_CROSSPLATFORMS}; do GOOS=${platform%/*} GOARCH=${platform##*/} go install std; done || true) \
    && go clean -cache

# Install g++, then download and install protoc for generating protobuf output
RUN apt-get update \
  && apt-get install -y rsync jq apt-utils file patch unzip \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Use dynamic cgo linking for architectures other than amd64 for the server platforms
# To install crossbuild essential for other architectures add the following repository.
# NOTE: we are not going to support cross-build from arm64 and ppc64le to other architectures
RUN echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu xenial main universe" > /etc/apt/sources.list.d/cgocrosscompiling.list \
  && apt-key adv --no-tty --keyserver keyserver.ubuntu.com --recv-keys 40976EAF437D05B5 3B4FE6ACC0B21F32 \
  && apt-get update \
  && apt-get install -y build-essential \
  && ([ "$(uname -m)" = "x86_64" ] && for platform in ${KUBE_DYNAMIC_CROSSPLATFORMS}; do apt-get install -y crossbuild-essential-${platform}; done || true) \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN PROTOBUF_VERSION=3.7.0; [ "$(uname -m)" = "x86_64" ] && ZIPNAME="protoc-${PROTOBUF_VERSION}-linux-x86_64.zip" || true; \
  [ "$(uname -m)" = "aarch64" ] && ZIPNAME="protoc-${PROTOBUF_VERSION}-linux-aarch_64.zip" || true; \
  [ "$(uname -m)" = "ppc64le" ] && ZIPNAME="protoc-${PROTOBUF_VERSION}-linux-ppcle_64.zip" || true; \
  mkdir /tmp/protoc && cd /tmp/protoc \
  && wget "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/${ZIPNAME}" \
  && unzip "${ZIPNAME}" \
  && chmod -R +rX /tmp/protoc \
  && cp -pr bin /usr/local \
  && cp -pr include /usr/local \
  && rm -rf /tmp/protoc \
  && protoc --version

# work around 64MB tmpfs size in Docker 1.6
ENV TMPDIR /tmp.k8s
RUN mkdir $TMPDIR \
  && chmod a+rwx $TMPDIR \
  && chmod o+t $TMPDIR

# Download and symlink etcd. We need this for our integration tests.
RUN export ETCD_VERSION=v3.2.24; \
  mkdir -p /usr/local/src/etcd \
  && cd /usr/local/src/etcd \
  && curl -fsSL https://github.com/coreos/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-linux-${ARCH}.tar.gz | tar -xz \
  && ln -s ../src/etcd/etcd-${ETCD_VERSION}-linux-${ARCH}/etcd /usr/local/bin/
