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
# to work with Kubernetes logging.

FROM cantara/alpine-openjdk-jdk8

MAINTAINER derjohn "himself@derjohn.de"

ENV LANG=C.UTF-8 \
    ES_VERSION=2.4.0

RUN apk upgrade --update

RUN apk add --update curl wget ca-certificates bash jq

RUN cd / && \
    curl -O https://download.elasticsearch.org/elasticsearch/release/org/elasticsearch/distribution/tar/elasticsearch/${ES_VERSION}/elasticsearch-${ES_VERSION}.tar.gz && \
    tar xf elasticsearch-${ES_VERSION}.tar.gz && \
    rm elasticsearch-${ES_VERSION}.tar.gz

RUN mkdir /data
RUN mkdir -p /elasticsearch-${ES_VERSION}/config/templates

COPY elasticsearch.yml /elasticsearch-${ES_VERSION}/config/elasticsearch.yml
COPY template-k8s-logstash.json /elasticsearch-${ES_VERSION}/config/templates/template-k8s-logstash.json
COPY elasticsearch_logging_discovery /elasticsearch-${ES_VERSION}/elasticsearch_logging_discovery
COPY run.sh /

VOLUME ["/data"]
EXPOSE 9200 9300

CMD ["/run.sh"]
