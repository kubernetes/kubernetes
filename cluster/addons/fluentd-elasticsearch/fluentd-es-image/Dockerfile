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

# This Dockerfile will build an image that is configured
# to run Fluentd with an Elasticsearch plug-in and the
# provided configuration file.
# TODO(a-robinson): Use a lighter base image, e.g. some form of busybox.
# The image acts as an executable for the binary /usr/sbin/td-agent.
# Note that fluentd is run with root permssion to allow access to
# log files with root only access under /var/log/containers/*
# Please see http://docs.fluentd.org/articles/install-by-deb for more
# information about installing fluentd using deb package.

FROM ubuntu:14.04
MAINTAINER Alex Robinson "arob@google.com"
MAINTAINER Jimmi Dyson "jimmidyson@gmail.com"

# Ensure there are enough file descriptors for running Fluentd.
RUN ulimit -n 65536

# Disable prompts from apt.
ENV DEBIAN_FRONTEND noninteractive

# Install prerequisites.
RUN apt-get update && \
    apt-get install -y -q curl make g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Fluentd.
RUN /usr/bin/curl -L https://td-toolbelt.herokuapp.com/sh/install-ubuntu-trusty-td-agent2.sh | sh

# Change the default user and group to root.
# Needed to allow access to /var/log/docker/... files.
RUN sed -i -e "s/USER=td-agent/USER=root/" -e "s/GROUP=td-agent/GROUP=root/" /etc/init.d/td-agent

# Install the Elasticsearch Fluentd plug-in.
RUN td-agent-gem install fluent-plugin-kubernetes_metadata_filter fluent-plugin-elasticsearch

# Copy the Fluentd configuration file.
COPY td-agent.conf /etc/td-agent/td-agent.conf

# Run the Fluentd service.
ENTRYPOINT ["td-agent"]
