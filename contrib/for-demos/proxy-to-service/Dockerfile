# Alpine linux would be great for this, but it's DNS does not use seach paths.
FROM progrium/busybox
MAINTAINER Tim Hockin "thockin@google.com"

RUN opkg-install socat

# Usage: docker run -p <host-port>:<port> <this-container> <tcp|udp> <port> <service-name>
ENTRYPOINT [ "sh", "-c", "PROTO=$(echo $0 | tr a-z A-Z); exec socat ${PROTO}-LISTEN:$1,reuseaddr,fork ${PROTO}:$2:$1" ]
