FROM busybox:ubuntu-14.04
MAINTAINER kyurtsever@google.com dengnan@google.com vmarmol@google.com jason@swindle.me

# Get cAdvisor binaries.
ADD http://storage.googleapis.com/cadvisor-bin/cadvisor-0.1.3 /usr/bin/cadvisor
RUN chmod +x /usr/bin/cadvisor

EXPOSE 8080
ENTRYPOINT ["/usr/bin/cadvisor"]
