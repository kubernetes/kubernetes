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

# TODO: get rid of bash dependency and switch to plain busybox.
# The tar in busybox also doesn't seem to understand compression.
FROM debian:jessie
MAINTAINER Prashanth.B <beeps@google.com>

# TODO: just use standard redis when there is one for 3.2.0.
RUN apt-get update && apt-get install -y wget make gcc

# See README.md
RUN wget -qO /redis-3.2.0.tar.gz http://download.redis.io/releases/redis-3.2.0.tar.gz && \
    tar -xzf /redis-3.2.0.tar.gz -C /tmp/ && rm /redis-3.2.0.tar.gz

# Clean out existing deps before installation
# see https://github.com/antirez/redis/issues/722
RUN cd /tmp/redis-3.2.0 && make distclean && mkdir -p /redis && \
    make install INSTALL_BIN=/redis && \
    mv /tmp/redis-3.2.0/redis.conf /redis/redis.conf && \
    rm -rf /tmp/redis-3.2.0

ADD on-start.sh /
# See contrib/pets/peer-finder for details
RUN wget -qO /peer-finder https://storage.googleapis.com/kubernetes-release/pets/peer-finder
ADD install.sh /
RUN chmod -c 755 /install.sh /on-start.sh /peer-finder
Entrypoint ["/install.sh"]
