FROM ubuntu:14.04

ENV GOLANG_VERSION 1.4rc1
ENV GOPATH /var/cache/drone
ENV GOROOT /usr/local/go
ENV PATH $PATH:$GOROOT/bin:$GOPATH/bin

ENV LANG C
ENV LC_ALL C

RUN apt-get update && apt-get install -y \
  wget ca-certificates git mercurial bzr \
  --no-install-recommends \
  && rm -rf /var/lib/apt/lists/*

RUN wget https://golang.org/dl/go$GOLANG_VERSION.linux-amd64.tar.gz --quiet && \
  tar -C /usr/local -xzf go$GOLANG_VERSION.linux-amd64.tar.gz && \
  rm go${GOLANG_VERSION}.linux-amd64.tar.gz

RUN go get github.com/axw/gocov/gocov github.com/mattn/goveralls github.com/golang/lint/golint
