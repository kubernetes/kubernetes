# docker build -t docker:simple -f Dockerfile.simple .
# docker run --rm docker:simple hack/make.sh dynbinary
# docker run --rm --privileged docker:simple hack/dind hack/make.sh test-unit
# docker run --rm --privileged -v /var/lib/docker docker:simple hack/dind hack/make.sh dynbinary test-integration-cli

# This represents the bare minimum required to build and test Docker.

FROM debian:jessie

# compile and runtime deps
# https://github.com/docker/docker/blob/master/project/PACKAGERS.md#build-dependencies
# https://github.com/docker/docker/blob/master/project/PACKAGERS.md#runtime-dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
		btrfs-tools \
		curl \
		gcc \
		git \
		golang \
		libdevmapper-dev \
		libsqlite3-dev \
		\
		ca-certificates \
		e2fsprogs \
		iptables \
		procps \
		xz-utils \
		\
		aufs-tools \
		lxc \
	&& rm -rf /var/lib/apt/lists/*

ENV AUTO_GOPATH 1
WORKDIR /usr/src/docker
COPY . /usr/src/docker
