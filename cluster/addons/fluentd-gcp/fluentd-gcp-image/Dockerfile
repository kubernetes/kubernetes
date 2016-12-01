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

MAINTAINER Mik Vyatskov "vmik@google.com"

# Copy binaries from the build output
COPY _output/jemalloc/lib /usr/lib/x86_64-linux-gnu
COPY _output/systemd/lib/libsystemd.* /usr/lib/x86_64-linux-gnu/
COPY _output/systemd/lib/libsystemd-journal.* /usr/lib/x86_64-linux-gnu/

COPY _output/ruby /opt/ruby

COPY _output/lib /usr/lib/x86_64-linux-gnu

# Copy the Fluentd configuration files for logging Docker container logs.
COPY fluent.conf /etc/fluent/fluent.conf
COPY agent.sh /agent.sh

# Start Fluentd to pick up our config that watches Docker container logs.
CMD /agent.sh $FLUENTD_ARGS
