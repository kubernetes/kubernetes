# This file describes the standard way to build Docker on s390x, using docker
#
# Usage:
#
# # Assemble the full dev environment. This is slow the first time.
# docker build -t docker -f Dockerfile.s390x .
#
# # Mount your source in an interactive container for quick testing:
# docker run -v `pwd`:/go/src/github.com/docker/docker --privileged -i -t docker bash
#
# # Run the test suite:
# docker run --privileged docker hack/make.sh test-unit test-integration-cli test-docker-py
#
# Note: AppArmor used to mess with privileged mode, but this is no longer
# the case. Therefore, you don't have to disable it anymore.
#

FROM s390x/debian:jessie

# Packaged dependencies
RUN apt-get update && apt-get install -y \
	apparmor \
	apt-utils \
	aufs-tools \
	automake \
	bash-completion \
	btrfs-tools \
	build-essential \
	cmake \
	createrepo \
	curl \
	dpkg-sig \
	git \
	iptables \
	jq \
	net-tools \
	libapparmor-dev \
	libcap-dev \
	libsystemd-journal-dev \
	libtool \
	mercurial \
	pkg-config \
	python-dev \
	python-mock \
	python-pip \
	python-websocket \
	xfsprogs \
	tar \
	vim-common \
	--no-install-recommends

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

# Get lvm2 source for compiling statically
ENV LVM2_VERSION 2.02.103
RUN mkdir -p /usr/local/lvm2 \
	&& curl -fsSL "https://mirrors.kernel.org/sourceware/lvm2/LVM2.${LVM2_VERSION}.tgz" \
		| tar -xzC /usr/local/lvm2 --strip-components=1
# See https://git.fedorahosted.org/cgit/lvm2.git/refs/tags for release tags

# Fix platform enablement in lvm2 to support s390x properly
RUN set -e \
	&& for f in config.guess config.sub; do \
		curl -fsSL -o "/usr/local/lvm2/autoconf/$f" "http://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=$f;hb=HEAD"; \
	done
# "arch.c:78:2: error: #error the arch code needs to know about your machine type"

# Compile and install lvm2
RUN cd /usr/local/lvm2 \
	&& ./configure \
		--build="$(gcc -print-multiarch)" \
		--enable-static_link \
	&& make device-mapper \
	&& make install_device-mapper
# See https://git.fedorahosted.org/cgit/lvm2.git/tree/INSTALL

# IMPORTANT: When updating this please note that stdlib archive/tar pkg is vendored
ENV GO_VERSION 1.8.3
RUN curl -fsSL "https://golang.org/dl/go${GO_VERSION}.linux-s390x.tar.gz" \
	| tar -xzC /usr/local

ENV PATH /go/bin:/usr/local/go/bin:$PATH
ENV GOPATH /go

# Dependency for golint
ENV GO_TOOLS_COMMIT 823804e1ae08dbb14eb807afc7db9993bc9e3cc3
RUN git clone https://github.com/golang/tools.git /go/src/golang.org/x/tools \
	&& (cd /go/src/golang.org/x/tools && git checkout -q $GO_TOOLS_COMMIT)

# Grab Go's lint tool
ENV GO_LINT_COMMIT 32a87160691b3c96046c0c678fe57c5bef761456
RUN git clone https://github.com/golang/lint.git /go/src/github.com/golang/lint \
	&& (cd /go/src/github.com/golang/lint && git checkout -q $GO_LINT_COMMIT) \
	&& go install -v github.com/golang/lint/golint

# Install two versions of the registry. The first is an older version that
# only supports schema1 manifests. The second is a newer version that supports
# both. This allows integration-cli tests to cover push/pull with both schema1
# and schema2 manifests.
ENV REGISTRY_COMMIT_SCHEMA1 ec87e9b6971d831f0eff752ddb54fb64693e51cd
ENV REGISTRY_COMMIT 47a064d4195a9b56133891bbb13620c3ac83a827
RUN set -x \
	&& export GOPATH="$(mktemp -d)" \
	&& git clone https://github.com/docker/distribution.git "$GOPATH/src/github.com/docker/distribution" \
	&& (cd "$GOPATH/src/github.com/docker/distribution" && git checkout -q "$REGISTRY_COMMIT") \
	&& GOPATH="$GOPATH/src/github.com/docker/distribution/Godeps/_workspace:$GOPATH" \
		go build -o /usr/local/bin/registry-v2 github.com/docker/distribution/cmd/registry \
	&& (cd "$GOPATH/src/github.com/docker/distribution" && git checkout -q "$REGISTRY_COMMIT_SCHEMA1") \
	&& GOPATH="$GOPATH/src/github.com/docker/distribution/Godeps/_workspace:$GOPATH" \
		go build -o /usr/local/bin/registry-v2-schema1 github.com/docker/distribution/cmd/registry \
	&& rm -rf "$GOPATH"

# Install notary and notary-server
ENV NOTARY_VERSION v0.5.0
RUN set -x \
	&& export GOPATH="$(mktemp -d)" \
	&& git clone https://github.com/docker/notary.git "$GOPATH/src/github.com/docker/notary" \
	&& (cd "$GOPATH/src/github.com/docker/notary" && git checkout -q "$NOTARY_VERSION") \
	&& GOPATH="$GOPATH/src/github.com/docker/notary/vendor:$GOPATH" \
		go build -o /usr/local/bin/notary-server github.com/docker/notary/cmd/notary-server \
	&& GOPATH="$GOPATH/src/github.com/docker/notary/vendor:$GOPATH" \
		go build -o /usr/local/bin/notary github.com/docker/notary/cmd/notary \
	&& rm -rf "$GOPATH"

# Get the "docker-py" source so we can run their integration tests
ENV DOCKER_PY_COMMIT a962578e515185cf06506050b2200c0b81aa84ef
RUN git clone https://github.com/docker/docker-py.git /docker-py \
	&& cd /docker-py \
	&& git checkout -q $DOCKER_PY_COMMIT \
	&& pip install -r test-requirements.txt

# Set user.email so crosbymichael's in-container merge commits go smoothly
RUN git config --global user.email 'docker-dummy@example.com'

# Add an unprivileged user to be used for tests which need it
RUN groupadd -r docker
RUN useradd --create-home --gid docker unprivilegeduser

VOLUME /var/lib/docker
WORKDIR /go/src/github.com/docker/docker
ENV DOCKER_BUILDTAGS apparmor selinux seccomp

# Let us use a .bashrc file
RUN ln -sfv $PWD/.bashrc ~/.bashrc

# Register Docker's bash completion.
RUN ln -sv $PWD/contrib/completion/bash/docker /etc/bash_completion.d/docker

# Get useful and necessary Hub images so we can "docker load" locally instead of pulling
COPY contrib/download-frozen-image-v2.sh /go/src/github.com/docker/docker/contrib/
RUN ./contrib/download-frozen-image-v2.sh /docker-frozen-images \
	s390x/buildpack-deps:jessie@sha256:552dec28146e4d2591fc0309aebdbac9e4fb1f335d90c70a14bbf72fb8bb1be5 \
	s390x/busybox:latest@sha256:e32f40c39ca596a4317392bd32809bb188c4ae5864ea827c3219c75c50069964 \
	s390x/debian:jessie@sha256:6994e3ffa5a1dabea09d536f350b3ed2715292cb469417c42a82b70fcbff7d32 \
	s390x/hello-world:latest@sha256:602db500fee63934292260e65c0c528128ad1c1c7c6497f95bbbac7d4d5312f1
# See also ensureFrozenImagesLinux() in "integration-cli/fixtures_linux_daemon_test.go" (which needs to be updated when adding images to this list)

# Install tomlv, vndr, runc, containerd, tini, docker-proxy
# Please edit hack/dockerfile/install-binaries.sh to update them.
COPY hack/dockerfile/binaries-commits /tmp/binaries-commits
COPY hack/dockerfile/install-binaries.sh /tmp/install-binaries.sh
RUN /tmp/install-binaries.sh tomlv vndr runc containerd tini proxy dockercli
ENV PATH=/usr/local/cli:$PATH

# Wrap all commands in the "docker-in-docker" script to allow nested containers
ENTRYPOINT ["hack/dind"]

# Upload docker source
COPY . /go/src/github.com/docker/docker
