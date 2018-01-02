# Based on docker-library's golang 1.6 alpine and wheezy docker files.
# https://github.com/docker-library/golang/blob/master/1.6/alpine/Dockerfile
# https://github.com/docker-library/golang/blob/master/1.6/wheezy/Dockerfile
FROM buildpack-deps:wheezy-scm

ENV GOLANG_SRC_REPO_URL https://github.com/golang/go

ENV GOLANG_BOOTSTRAP_URL https://storage.googleapis.com/golang/go1.4.3.linux-amd64.tar.gz
ENV GOLANG_BOOTSTRAP_SHA256 ce3140662f45356eb78bc16a88fc7cfb29fb00e18d7c632608245b789b2086d2
ENV GOLANG_BOOTSTRAP_PATH /usr/local/bootstrap

# gcc for cgo
RUN apt-get update && apt-get install -y --no-install-recommends \
		g++ \
		gcc \
		libc6-dev \
		make \
		git \
	&& rm -rf /var/lib/apt/lists/*

# Setup the Bootstrap
RUN mkdir -p "$GOLANG_BOOTSTRAP_PATH" \
	&& curl -fsSL "$GOLANG_BOOTSTRAP_URL" -o golang.tar.gz \
	&& echo "$GOLANG_BOOTSTRAP_SHA256  golang.tar.gz" | sha256sum -c - \
	&& tar -C "$GOLANG_BOOTSTRAP_PATH" -xzf golang.tar.gz \
	&& rm golang.tar.gz

# Get and build Go tip
RUN export GOROOT_BOOTSTRAP=$GOLANG_BOOTSTRAP_PATH/go \
	&& git clone "$GOLANG_SRC_REPO_URL" /usr/local/go \
	&& cd /usr/local/go/src \
	&& ./make.bash \
	&& rm -rf "$GOLANG_BOOTSTRAP_PATH" /usr/local/go/pkg/bootstrap 

# Build Go workspace and environment
ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH
RUN mkdir -p "$GOPATH/src" "$GOPATH/bin" \
	&& chmod -R 777 "$GOPATH"

WORKDIR $GOPATH
