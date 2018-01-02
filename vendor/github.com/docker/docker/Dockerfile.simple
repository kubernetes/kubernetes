# docker build -t docker:simple -f Dockerfile.simple .
# docker run --rm docker:simple hack/make.sh dynbinary
# docker run --rm --privileged docker:simple hack/dind hack/make.sh test-unit
# docker run --rm --privileged -v /var/lib/docker docker:simple hack/dind hack/make.sh dynbinary test-integration-cli

# This represents the bare minimum required to build and test Docker.

FROM debian:jessie

# allow replacing httpredir or deb mirror
ARG APT_MIRROR=deb.debian.org
RUN sed -ri "s/(httpredir|deb).debian.org/$APT_MIRROR/g" /etc/apt/sources.list

# Compile and runtime deps
# https://github.com/docker/docker/blob/master/project/PACKAGERS.md#build-dependencies
# https://github.com/docker/docker/blob/master/project/PACKAGERS.md#runtime-dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
		btrfs-tools \
		build-essential \
		curl \
		cmake \
		gcc \
		git \
		libapparmor-dev \
		libdevmapper-dev \
		ca-certificates \
		e2fsprogs \
		iptables \
		procps \
		xfsprogs \
		xz-utils \
		\
		aufs-tools \
		vim-common \
	&& rm -rf /var/lib/apt/lists/*

# Install seccomp: the version shipped upstream is too old
ENV SECCOMP_VERSION 2.3.2
RUN set -x \
	&& export SECCOMP_PATH="$(mktemp -d)" \
	&& curl -fsSL "https://github.com/seccomp/libseccomp/releases/download/v${SECCOMP_VERSION}/libseccomp-${SECCOMP_VERSION}.tar.gz" \
		| tar -xzC "$SECCOMP_PATH" --strip-components=1 \
	&& ( \
		cd "$SECCOMP_PATH" \
		&& ./configure --prefix=/usr/local \
		&& make \
		&& make install \
		&& ldconfig \
	) \
	&& rm -rf "$SECCOMP_PATH"

# Install Go
# IMPORTANT: If the version of Go is updated, the Windows to Linux CI machines
#            will need updating, to avoid errors. Ping #docker-maintainers on IRC
#            with a heads-up.
# IMPORTANT: When updating this please note that stdlib archive/tar pkg is vendored
ENV GO_VERSION 1.8.3
RUN curl -fsSL "https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz" \
	| tar -xzC /usr/local
ENV PATH /go/bin:/usr/local/go/bin:$PATH
ENV GOPATH /go
ENV CGO_LDFLAGS -L/lib

# Install runc, containerd, tini and docker-proxy
# Please edit hack/dockerfile/install-binaries.sh to update them.
COPY hack/dockerfile/binaries-commits /tmp/binaries-commits
COPY hack/dockerfile/install-binaries.sh /tmp/install-binaries.sh
RUN /tmp/install-binaries.sh runc containerd tini proxy dockercli
ENV PATH=/usr/local/cli:$PATH

ENV AUTO_GOPATH 1
WORKDIR /usr/src/docker
COPY . /usr/src/docker
