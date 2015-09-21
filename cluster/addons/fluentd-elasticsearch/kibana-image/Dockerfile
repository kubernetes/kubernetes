# A Dockerfile for creating a Kibana container that is designed
# to work with Kubernetes logging.

FROM java:openjdk-7-jre
MAINTAINER Satnam Singh "satnam@google.com"

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean

RUN cd / && \
    curl -O https://download.elastic.co/kibana/kibana/kibana-4.0.2-linux-x64.tar.gz && \
    tar xf kibana-4.0.2-linux-x64.tar.gz && \
    rm kibana-4.0.2-linux-x64.tar.gz

COPY run.sh /run.sh

EXPOSE 5601
CMD ["/run.sh"]
