FROM alpine:3.2

ENV PATH /go/bin:/usr/local/go/bin:$PATH
ENV GOPATH /go
ENV USER root

COPY . /go/src/github.com/cloudflare/cfssl

RUN buildDeps=' \
		go \
		git \
		gcc \
		libc-dev \
		libtool \
		libgcc \
	' \
	set -x && \
	apk update && \
	apk add $buildDeps && \
	cd /go/src/github.com/cloudflare/cfssl && \
	go get github.com/GeertJohan/go.rice/rice && rice embed-go -i=./cli/serve && \
	cp -R /go/src/github.com/cloudflare/cfssl/vendor/github.com/cloudflare/cfssl_trust /etc/cfssl && \
	go build -o /usr/bin/cfssl ./cmd/cfssl && \
	go build -o /usr/bin/cfssljson ./cmd/cfssljson && \
	go build -o /usr/bin/mkbundle ./cmd/mkbundle && \
	go build -o /usr/bin/multirootca ./cmd/multirootca && \
	apk del $buildDeps && \
	rm -rf /var/cache/apk/* && \
	rm -rf /go && \
	echo "Build complete."


VOLUME [ "/etc/cfssl" ]
WORKDIR /etc/cfssl

EXPOSE 8888

ENTRYPOINT ["cfssl"]
CMD ["--help"]
