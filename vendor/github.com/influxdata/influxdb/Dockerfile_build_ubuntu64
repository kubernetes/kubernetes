FROM ubuntu:trusty

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python-software-properties \
    software-properties-common \
    wget \
    git \
    mercurial \
    make \
    ruby \
    ruby-dev \
    rpm \
    zip \
    python \
    python-boto \
    asciidoc \
    xmlto \
    docbook-xsl

RUN gem install fpm

# Install go
ENV GOPATH /root/go
ENV GO_VERSION 1.7.4
ENV GO_ARCH amd64
RUN wget https://storage.googleapis.com/golang/go${GO_VERSION}.linux-${GO_ARCH}.tar.gz; \
   tar -C /usr/local/ -xf /go${GO_VERSION}.linux-${GO_ARCH}.tar.gz ; \
   rm /go${GO_VERSION}.linux-${GO_ARCH}.tar.gz
ENV PATH /usr/local/go/bin:$PATH

ENV PROJECT_DIR $GOPATH/src/github.com/influxdata/influxdb
ENV PATH $GOPATH/bin:$PATH
RUN mkdir -p $PROJECT_DIR
WORKDIR $PROJECT_DIR

VOLUME $PROJECT_DIR

ENTRYPOINT [ "/root/go/src/github.com/influxdata/influxdb/build.py" ]
