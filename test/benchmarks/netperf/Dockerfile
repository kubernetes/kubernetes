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

# network performance tests in containers
#
# Run as the Orchestrator: Arguments: --mode=orchestrator
#
# Run as a Worker (first get orchestrator virtual IP address):
# kubectl get svc --format "{{ .NetworkSettings.IPAddress }}" netperf-orch
#
# Args: --mode=worker --host=<service cluster ip> --port=5202
#
FROM debian:jessie

MAINTAINER Girish Kalele <gkalele@google.com>
# install binary and remove cache
RUN apt-get update \
    && apt-get install -y iperf3 curl wget net-tools gcc make \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /tmp

# Download and build netperf from sources
RUN curl -LO ftp://ftp.netperf.org/netperf/netperf-2.7.0.tar.gz && tar -xzf netperf-2.7.0.tar.gz
RUN cd netperf-2.7.0 && ./configure --prefix=/usr/local --bindir /usr/local/bin && make && make install

COPY nptests /usr/bin/

ENTRYPOINT ["nptests"]
