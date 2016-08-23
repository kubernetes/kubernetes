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
# to use Fluentd to collect all Docker container log files
# and then cause them to be ingested using the Google Cloud
# Logging API. This configuration assumes that the host performning
# the collection is a VM that has been created with a logging.write
# scope and that the Logging API has been enabled for the project
# in the Google Developer Console.

FROM gcr.io/google_containers/ubuntu-slim:0.4
MAINTAINER Alex Robinson "arob@google.com"

# Disable prompts from apt.
ENV DEBIAN_FRONTEND noninteractive
# Keeps unneeded configs from being installed along with fluentd.
ENV DO_NOT_INSTALL_CATCH_ALL_CONFIG true

RUN apt-get -q update && \
    apt-get install -y curl ca-certificates gcc make bash && \
    apt-get install -y --reinstall lsb-base lsb-release && \
    echo "Installing logging agent" && \
    curl -sSL https://dl.google.com/cloudagents/install-logging-agent.sh | bash && \
    /usr/sbin/google-fluentd-gem install fluent-plugin-record-reformer -v 0.8.1  && \
    /usr/sbin/google-fluentd-gem install fluent-plugin-systemd -v 0.0.3 && \
    apt-get remove -y gcc make && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    /opt/google-fluentd/embedded/share/doc \
    /opt/google-fluentd/embedded/share/gtk-doc \
    /opt/google-fluentd/embedded/lib/postgresql \
    /opt/google-fluentd/embedded/bin/postgres \
    /opt/google-fluentd/embedded/share/postgresql \
    /var/log/google-fluentd

# Copy the Fluentd configuration files for logging Docker container logs.
# Either configuration file can be used by specifying `-c <file>` as a command
# line argument.
COPY google-fluentd.conf /etc/google-fluentd/google-fluentd.conf
COPY google-fluentd-journal.conf /etc/google-fluentd/google-fluentd-journal.conf

# Start Fluentd to pick up our config that watches Docker container logs.
CMD /usr/sbin/google-fluentd "$FLUENTD_ARGS"
