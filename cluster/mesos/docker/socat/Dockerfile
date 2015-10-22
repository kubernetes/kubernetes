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
