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

# This file creates a standard build environment for building Kubernetes
ARG KUBE_CROSS_IMAGE
ARG KUBE_CROSS_VERSION

FROM ${KUBE_CROSS_IMAGE}:${KUBE_CROSS_VERSION}

# Mark this as a kube-build container
RUN touch /kube-build-image

# To run as non-root we sometimes need to rebuild go stdlib packages.
RUN chmod -R a+rwx /usr/local/go/pkg

# For running integration tests /var/run/kubernetes is required
# and should be writable by user
RUN mkdir /var/run/kubernetes && chmod a+rwx /var/run/kubernetes

# The kubernetes source is expected to be mounted here.  This will be the base
# of operations.
ENV HOME=/go/src/k8s.io/kubernetes
WORKDIR ${HOME}

# Make output from the dockerized build go someplace else
ENV KUBE_OUTPUT_SUBPATH=_output/dockerized

# Pick up version stuff here as we don't copy our .git over.
ENV KUBE_GIT_VERSION_FILE=${HOME}/.dockerized-kube-version-defs

# Add system-wide git user information
RUN git config --system user.email "nobody@k8s.io" \
  && git config --system user.name "kube-build-image"

# Fix permissions on gopath
RUN chmod -R a+rwx $GOPATH

# Make log messages use the right timezone
ADD localtime /etc/localtime
RUN chmod a+r /etc/localtime

# Set up rsyncd
ADD rsyncd.password /
RUN chmod a+r /rsyncd.password
ADD rsyncd.sh /
RUN chmod a+rx /rsyncd.sh
