FROM busybox
MAINTAINER Tim Hockin <thockin@google.com>
ADD serve_hostname /serve_hostname
ADD serve_hostname.go /serve_hostname.go
EXPOSE 9376
ENTRYPOINT ["/serve_hostname"]
