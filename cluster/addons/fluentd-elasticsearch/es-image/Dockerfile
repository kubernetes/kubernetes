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

# A Dockerfile for creating an Elasticsearch instance that is designed
# to work with Kubernetes logging. Inspired by the Dockerfile
# dockerfile/elasticsearch

FROM java:openjdk-8-jre


ENV DEBIAN_FRONTEND noninteractive
ENV ELASTICSEARCH_VERSION 2.4.1

RUN apt-get update \
    && apt-get install -y curl \
    && apt-get clean

RUN set -x \
    && cd / \
    && mkdir /elasticsearch \
    && curl -O https://download.elastic.co/elasticsearch/release/org/elasticsearch/distribution/tar/elasticsearch/$ELASTICSEARCH_VERSION/elasticsearch-$ELASTICSEARCH_VERSION.tar.gz \
    && tar xf elasticsearch-$ELASTICSEARCH_VERSION.tar.gz -C /elasticsearch --strip-components=1 \
    && rm elasticsearch-$ELASTICSEARCH_VERSION.tar.gz

RUN mkdir -p /elasticsearch/config/templates
COPY template-k8s-logstash.json /elasticsearch/config/templates/template-k8s-logstash.json

COPY config /elasticsearch/config

COPY run.sh /
COPY elasticsearch_logging_discovery /

RUN useradd --no-create-home --user-group elasticsearch \
    && mkdir /data \
    && chown -R elasticsearch:elasticsearch /elasticsearch

VOLUME ["/data"]
EXPOSE 9200 9300

CMD /run.sh