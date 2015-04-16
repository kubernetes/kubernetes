# A Dockerfile for creating an Elasticsearch instance that is designed
# to work with Kubernetes logging. Inspired by the Dockerfile
# dockerfile/elasticsearch

FROM java:openjdk-7-jre
MAINTAINER Satnam Singh "satnam@google.com"

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean

RUN cd / && \
    curl -O https://download.elasticsearch.org/elasticsearch/elasticsearch/elasticsearch-1.4.4.tar.gz && \
    tar xf elasticsearch-1.4.4.tar.gz && \
    mv elasticsearch-1.4.4 /elasticsearch && \
    rm -rf elasticsearch-1.4.4.tar.gz

ADD elasticsearch.yml /elasticsearch/config/elasticsearch.yml

VOLUME ["/data"]
WORKDIR /data
CMD ["/elasticsearch/bin/elasticsearch"]

EXPOSE 9200
EXPOSE 9300
