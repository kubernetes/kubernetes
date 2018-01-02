FROM alpine:3.2
MAINTAINER vishnuk@google.com

ENV GLIBC_VERSION "2.23-r1"

RUN apk add --update ca-certificates curl && \
    curl -Ls https://github.com/andyshinn/alpine-pkg-glibc/releases/download/${GLIBC_VERSION}/glibc-${GLIBC_VERSION}.apk -o glibc-${GLIBC_VERSION}.apk && \
    curl -Ls https://github.com/andyshinn/alpine-pkg-glibc/releases/download/${GLIBC_VERSION}/glibc-bin-${GLIBC_VERSION}.apk -o glibc-bin-${GLIBC_VERSION}.apk && \
    apk add --allow-untrusted glibc-${GLIBC_VERSION}.apk glibc-bin-${GLIBC_VERSION}.apk 

RUN for cert in `ls -1 /etc/ssl/certs/*.crt | grep -v /etc/ssl/certs/ca-certificates.crt`; do cat "$cert" >> /etc/ssl/certs/ca-certificates.crt; done

# cAdvisor discovery via external files.
VOLUME /var/run/heapster/hosts
ADD heapster /heapster
ADD eventer /eventer

ENTRYPOINT ["/heapster"]
