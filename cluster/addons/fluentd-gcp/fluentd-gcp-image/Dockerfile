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

# This Dockerfile will build an image that is configured
# to use Fluentd to collect all Docker container log files
# and then cause them to be ingested using the Google Cloud
# Logging API. This configuration assumes that the host performning
# the collection is a VM that has been created with a logging.write
# scope and that the Logging API has been enabled for the project
# in the Google Developer Console. 

FROM gcr.io/google_containers/ubuntu-slim:0.2
MAINTAINER Alex Robinson "arob@google.com"

# Disable prompts from apt.
ENV DEBIAN_FRONTEND noninteractive
# Keeps unneeded configs from being installed along with fluentd.
ENV DO_NOT_INSTALL_CATCH_ALL_CONFIG true

# --reinstall is required to install lsb-base (required by install-logging-agent.sh)
RUN apt-get update && apt-get install --reinstall -y --no-install-recommends \
  curl ca-certificates \
  bash \
  lsb-base \
  lsb-release && \
  curl -sSL https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
  curl -sSL https://dl.google.com/cloudagents/install-logging-agent.sh | bash && \
  apt-get autoremove -y && \
  apt-get clean -y && \
  rm -rf /var/lib/apt/lists/* \
  /opt/google-fluentd/embedded/share/doc \
  /opt/google-fluentd/embedded/share/gtk-doc \
  /opt/google-fluentd/embedded/lib/postgresql \
  /opt/google-fluentd/embedded/bin/postgres \
  /opt/google-fluentd/embedded/share/postgresql \
  /var/log/google-fluentd

# Install the record reformer plugin.
# Remove the misleading log file that gets generated when the agent is installed
RUN /usr/sbin/google-fluentd-gem install fluent-plugin-record-reformer

# Copy the Fluentd configuration file for logging Docker container logs.
COPY google-fluentd.conf /etc/google-fluentd/google-fluentd.conf

# Start Fluentd to pick up our config that watches Docker container logs.
CMD /usr/sbin/google-fluentd "$FLUENTD_ARGS"
