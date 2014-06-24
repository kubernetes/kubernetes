FROM ubuntu
MAINTAINER https://github.com/GoogleCloudPlatform/kubernetes

RUN  apt-get update && apt-get install -yq curl git mercurial

ENV  GO_VERSION 1.2
RUN  curl -s https://go.googlecode.com/files/go${GO_VERSION}.linux-amd64.tar.gz | tar -C /usr/local -xzf -
ENV  GOPATH  /go
ENV  PATH    $GOPATH/bin:/usr/local/go/bin:$PATH

ADD     . $GOPATH/src/github.com/GoogleCloudPlatform/kubernetes
WORKDIR /go/src/github.com/GoogleCloudPlatform/kubernetes/cmd

RUN set -e; for c in *; do ( cd $c && go get -d && go build -o /usr/bin/$c ); done
