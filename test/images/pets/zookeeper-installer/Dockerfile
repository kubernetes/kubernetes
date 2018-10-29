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
FROM BASEIMAGE

CROSS_BUILD_COPY qemu-QEMUARCH-static /usr/bin/

RUN clean-install wget netcat

ADD on-start.sh /

COPY peer-finder /

# See README.md
RUN wget -q -O /zookeeper-3.5.0-alpha.tar.gz http://archive.apache.org/dist/zookeeper/zookeeper-3.5.0-alpha/zookeeper-3.5.0-alpha.tar.gz && \
    tar -xzf /zookeeper-3.5.0-alpha.tar.gz -C /tmp/ && mv /tmp/zookeeper-3.5.0-alpha /zookeeper && rm /zookeeper-3.5.0-alpha.tar.gz

ADD install.sh /
RUN chmod -c 755 /install.sh /on-start.sh /peer-finder
Entrypoint ["/install.sh"]
