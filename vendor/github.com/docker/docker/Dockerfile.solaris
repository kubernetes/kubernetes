# Defines an image that hosts a native Docker build environment for Solaris
# TODO: Improve stub

FROM solaris:latest

# compile and runtime deps
RUN pkg install --accept \
		git \
		gnu-coreutils \
		gnu-make \
		gnu-tar \
		diagnostic/top \
		golang \
		library/golang/* \
		developer/gcc-*

ENV GOPATH /go/:/usr/lib/gocode/1.5/
WORKDIR /go/src/github.com/docker/docker
COPY . /go/src/github.com/docker/docker
