FROM ubuntu:16.04

# Use the most recent ubuntu sources
RUN echo 'deb http://en.archive.ubuntu.com/ubuntu/ artful main universe' >> /etc/apt/sources.list

# Dependencies to get the git sources and go binaries
RUN apt-get update && apt-get install -y \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Get the git sources. If not cached, this takes O(5 minutes).
WORKDIR /git
RUN git config --global advice.detachedHead false
# Linux Kernel: Released 03 Sep 2017
RUN git clone --branch v4.13 --depth 1 https://kernel.googlesource.com/pub/scm/linux/kernel/git/torvalds/linux
# GNU C library: Released 02 Aug 2017 (we should try to get a secure way to clone this)
RUN git clone --branch glibc-2.26 --depth 1 git://sourceware.org/git/glibc.git

# Get Go 1.9.2
ENV GOLANG_VERSION 1.9.2
ENV GOLANG_DOWNLOAD_URL https://golang.org/dl/go$GOLANG_VERSION.linux-amd64.tar.gz
ENV GOLANG_DOWNLOAD_SHA256 de874549d9a8d8d8062be05808509c09a88a248e77ec14eb77453530829ac02b

RUN curl -fsSL "$GOLANG_DOWNLOAD_URL" -o golang.tar.gz \
    && echo "$GOLANG_DOWNLOAD_SHA256  golang.tar.gz" | sha256sum -c - \
    && tar -C /usr/local -xzf golang.tar.gz \
    && rm golang.tar.gz

ENV PATH /usr/local/go/bin:$PATH

# Linux and Glibc build dependencies
RUN apt-get update && apt-get install -y \
        gawk make python \
        gcc gcc-multilib \
        gettext texinfo \
    && rm -rf /var/lib/apt/lists/*
# Emulator and cross compilers
RUN apt-get update && apt-get install -y \
        qemu \
        gcc-aarch64-linux-gnu       gcc-arm-linux-gnueabi     \
        gcc-mips-linux-gnu          gcc-mips64-linux-gnuabi64 \
        gcc-mips64el-linux-gnuabi64 gcc-mipsel-linux-gnu      \
        gcc-powerpc64-linux-gnu     gcc-powerpc64le-linux-gnu \
        gcc-s390x-linux-gnu         gcc-sparc64-linux-gnu     \
    && rm -rf /var/lib/apt/lists/*

# Let the scripts know they are in the docker environment
ENV GOLANG_SYS_BUILD docker
WORKDIR /build
ENTRYPOINT ["go", "run", "linux/mkall.go", "/git/linux", "/git/glibc"]
