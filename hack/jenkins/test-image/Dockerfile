# Copyright 2015 The Kubernetes Authors.
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

# This file creates a build environment for building and running kubernetes
# unit and integration tests

FROM golang:1.6.3
MAINTAINER  Jeff Lowdermilk <jeffml@google.com>

# Setup workspace and symlink to gopath
WORKDIR /workspace
RUN mkdir -p /go/src/k8s.io/kubernetes /workspace \
    && ln -s /go/src/k8s.io/kubernetes /workspace/kubernetes
ENV WORKSPACE=/workspace \
    TERM=xterm

# Install linux packages
# bc is needed by shell2junit
# dnsutils is needed by federation cluster scripts.
# file is used when uploading test artifacts to GCS.
# jq is used by hack/verify-godep-licenses.sh
# python-pip is needed to install the AWS cli.
# netcat is used by integration test scripts.
RUN apt-get -o Acquire::Check-Valid-Until=false update && apt-get install -y \
        bc \
	dnsutils \
	file \
	jq \
	python-pip \
	netcat-openbsd \
	rsync \
	--no-install-recommends \
	&& rm -rf /var/lib/apt/lists/*

# Install docker
# Note: 1.11+ changes the tarball format
RUN curl -L "https://get.docker.com/builds/Linux/x86_64/docker-1.9.1.tgz" \
    | tar -C /usr/bin -xvzf- --strip-components=3 usr/local/bin/docker

# Install any go packages
# TODO(fejta): migrate this to a unit test image
RUN go get \
        github.com/golang/lint/golint

