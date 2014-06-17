# Copyright 2014 Google Inc. All rights reserved.
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

# This file creates a standard build environment for building Kubernetes

FROM  google/debian:wheezy
MAINTAINER  Joe Beda <jbeda@google.com>

RUN apt-get update -y && apt-get install --no-install-recommends -y -q \
  curl \
  build-essential \
  ca-certificates \
  git \
  mercurial \
  rsync

# Install Go
# TODO(jbeda) -- we need to verify this against the hash
RUN curl -s https://storage.googleapis.com/golang/go1.2.2.src.tar.gz | tar -C /usr/local -xz
ENV PATH  /usr/local/go/bin:$PATH
RUN cd /usr/local/go/src && ./make.bash --no-clean 2>&1

# Compile Go for cross compilation
ENV KUBE_CROSSPLATFORMS \
  linux/386 linux/arm \
  darwin/amd64 darwin/386
# (set an explicit GOARM of 5 for maximum compatibility)
ENV GOARM 5
RUN cd /usr/local/go/src && \
  bash -xc 'for platform in $KUBE_CROSSPLATFORMS; do GOOS=${platform%/*} GOARCH=${platform##*/} ./make.bash --no-clean 2>&1; done'

# Set up Go Environment
ENV PATH    /go/bin:$PATH
ENV GOPATH  /go:/go/src/github.com/GoogleCloudPlatform/kubernetes/third_party
ENV GOOS    linux
ENV GOARCH  amd64

# Get the code coverage tool and etcd for integration tests
RUN go get code.google.com/p/go.tools/cmd/cover github.com/coreos/etcd

# Mark this as a kube-build container
RUN touch /kube-build-image

WORKDIR /go/src/github.com/GoogleCloudPlatform/kubernetes

# Upload Kubernetes
ADD kube-source.tar.gz /go/src/github.com/GoogleCloudPlatform/kubernetes
