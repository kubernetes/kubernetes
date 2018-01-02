FROM golang:1.6.0
MAINTAINER vishnuk@google.com

RUN apt-get install -y git
RUN git clone https://github.com/kubernetes/heapster.git /go/src/k8s.io/heapster

RUN cd /go/src/k8s.io/heapster && make && mv heapster /heapster && mv eventer /eventer

ENTRYPOINT ["/heapster"]
