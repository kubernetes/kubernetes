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

FROM google/debian:jessie

COPY cassandra.list /etc/apt/sources.list.d/cassandra.list
COPY run.sh /run.sh

RUN gpg --keyserver pgp.mit.edu --recv-keys F758CE318D77295D && \
  gpg --export --armor F758CE318D77295D | apt-key add - && \
  gpg --keyserver pgp.mit.edu --recv-keys 2B5C1B00 && \
  gpg --export --armor 2B5C1B00 | apt-key add - && \
  gpg --keyserver pgp.mit.edu --recv-keys 0353B12C && \
  gpg --export --armor 0353B12C | apt-key add - && \
  apt-get update && \
  apt-get -qq -y install procps  cassandra openjdk-8-jre-headless && \
  chmod a+rx /run.sh && \
  mkdir -p /cassandra_data/data && \
  chown -R cassandra.cassandra /etc/cassandra /cassandra_data && \
  chmod o+w -R /etc/cassandra /cassandra_data && \
  rm -rf /var/lib/apt/lists/* && \
  rm -rf /usr/share/doc/ && \
  rm -rf /usr/share/doc-base/ && \
  rm -rf /usr/share/man/ && \
  rm -rf /tmp/* 

COPY cassandra.yaml /etc/cassandra/cassandra.yaml
COPY logback.xml /etc/cassandra/logback.xml
COPY kubernetes-cassandra.jar /kubernetes-cassandra.jar

VOLUME ["/cassandra_data/data"]    

USER cassandra

CMD /run.sh
