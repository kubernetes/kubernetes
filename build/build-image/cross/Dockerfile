# Copyright 2016 The Kubernetes Authors All rights reserved.
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

FROM  golang:1.4.2
MAINTAINER  Joe Beda <jbeda@google.com>

ENV KUBE_CROSSPLATFORMS \
  linux/386 linux/arm \
  darwin/amd64 darwin/386 \
  windows/amd64 windows/386

RUN cd /usr/src/go/src && for platform in ${KUBE_CROSSPLATFORMS}; do GOOS=${platform%/*} GOARCH=${platform##*/} ./make.bash --no-clean; done

# Install g++, then download and install protoc for generating protobuf output
RUN apt-get install -y g++ && apt-get clean && rm -rf /var/lib/apt/lists/* &&\
    mkdir -p /usr/local/src/protobuf && cd /usr/local/src/protobuf &&\
    wget -q https://github.com/google/protobuf/releases/download/v3.0.0-beta-2/protobuf-cpp-3.0.0-beta-2.tar.gz &&\
    tar xzvf protobuf-cpp-3.0.0-beta-2.tar.gz &&\
    cd protobuf-3.0.0-beta-2 &&\
    ./configure &&\
    make install &&\
    ldconfig &&\
    cd .. &&\
    rm -rf protobuf-3.0.0-beta-2 &&\
    protoc --version

