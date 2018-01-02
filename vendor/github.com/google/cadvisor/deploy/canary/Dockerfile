FROM golang:latest
MAINTAINER vmarmol@google.com

RUN apt-get update && apt-get install -y git dmsetup && apt-get clean
RUN git clone https://github.com/google/cadvisor.git /go/src/github.com/google/cadvisor
RUN cd /go/src/github.com/google/cadvisor && make

ENTRYPOINT ["/go/src/github.com/google/cadvisor/cadvisor"]

