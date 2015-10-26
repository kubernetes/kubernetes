FROM busybox
MAINTAINER Abhishek Shah "abshah@google.com"

ADD netexec netexec
ADD netexec.go netexec.go
EXPOSE 8080
EXPOSE 8081

RUN mkdir /uploads

ENTRYPOINT ["/netexec"]
