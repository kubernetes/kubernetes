# Copyright 2015 Google Inc. All rights reserved.
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

# This file creates a standard build environment for building the Kubernetes WebUI

FROM node:0.12.3
MAINTAINER Jack Greenfield <jackgr@google.com>

# work around 64MB tmpfs size in Docker 1.6
ENV TMPDIR /tmp.k8s-www
RUN mkdir $TMPDIR

# Mark this as a kube-build-www container
RUN touch /kube-build-www-image

WORKDIR /go/src/github.com/GoogleCloudPlatform/kubernetes

# Propagate the git tree version into the build image
ADD kube-version-defs /kube-version-defs
ENV KUBE_GIT_VERSION_FILE /kube-version-defs

# Make output from the dockerized build go someplace else
ENV KUBE_OUTPUT_SUBPATH _output/dockerized

# Upload Kubernetes WebUI source
ADD kube-www-source.tar.gz /go/src/github.com/GoogleCloudPlatform/kubernetes

