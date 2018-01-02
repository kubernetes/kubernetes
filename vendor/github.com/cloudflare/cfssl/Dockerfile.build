FROM golang:1.6

ENV USER root

WORKDIR /go/src/github.com/cloudflare/cfssl
COPY . .

# restore all deps and build
RUN go get github.com/mitchellh/gox

ENTRYPOINT ["gox"]
