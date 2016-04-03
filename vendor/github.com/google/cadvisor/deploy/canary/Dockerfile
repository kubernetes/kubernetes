FROM golang:latest
MAINTAINER vmarmol@google.com

RUN apt-get install -y git dmsetup
RUN git clone https://github.com/google/cadvisor.git /go/src/github.com/google/cadvisor
RUN go get github.com/tools/godep
RUN cd /go/src/github.com/google/cadvisor && godep go build .

ENTRYPOINT ["/go/src/github.com/google/cadvisor/cadvisor"]

