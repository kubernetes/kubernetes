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

FROM ubuntu:14.04.3
MAINTAINER Mesosphere <support@mesosphere.io>

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -qqy \
        build-essential curl \
        && \
    apt-get clean

RUN mkdir -p /src
WORKDIR /src
RUN curl -f -osocat-1.7.2.4.tar.bz2 http://www.dest-unreach.org/socat/download/socat-1.7.2.4.tar.bz2
RUN tar -xjvf socat-1.7.2.4.tar.bz2 && cd socat-1.7.2.4 && ./configure --disable-openssl && LDFLAGS=-static make

VOLUME ["/target"]
CMD ["cp", "/src/socat-1.7.2.4/socat", "/target"]
