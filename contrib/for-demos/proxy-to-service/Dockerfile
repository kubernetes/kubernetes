# Alpine linux would be great for this, but it's DNS does not use search paths.
FROM progrium/busybox
MAINTAINER Tim Hockin "thockin@google.com"

RUN opkg-install socat
ADD start.sh start.sh

# Usage: docker run -p <host-port>:<port> <this-container> <tcp|udp> <port> <service-name> [timeout]
ENTRYPOINT ["/start.sh"]
