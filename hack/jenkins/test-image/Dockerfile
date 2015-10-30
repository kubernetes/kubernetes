# Copyright 2015 The Kubernetes Authors All rights reserved.
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

FROM golang:1.4
MAINTAINER  Jeff Lowdermilk <jeffml@google.com>

ENV WORKSPACE               /workspace
ENV TERM                    xterm

WORKDIR /workspace

RUN apt-get -o Acquire::Check-Valid-Until=false update && apt-get install -y rsync
RUN apt-get install -y file
RUN mkdir -p /go/src/k8s.io/kubernetes
RUN ln -s /go/src/k8s.io/kubernetes /workspace/kubernetes

RUN /bin/bash
