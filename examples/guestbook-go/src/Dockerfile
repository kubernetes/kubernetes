FROM google/golang:latest

RUN go get github.com/Gurpartap/guestbook-example && \
    cp /gopath/src/github.com/Gurpartap/guestbook-example/busybox-image/Dockerfile /gopath

CMD docker build --rm --force-rm -t gurpartap/guestbook-example /gopath
