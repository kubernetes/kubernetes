FROM ubuntu:14.04.3
MAINTAINER Mesosphere <support@mesosphere.io>

RUN locale-gen en_US.UTF-8
RUN dpkg-reconfigure locales
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -qqy \
        ca-certificates \
        wget \
        curl \
        && \
    apt-get clean

RUN curl -o- https://raw.githubusercontent.com/karlkfi/resolveip/v1.0.2/install.sh | bash

COPY ./bin/* /usr/local/bin/
COPY ./opt/* /opt/
