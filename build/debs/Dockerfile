FROM golang:1.12

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update -y \
    && apt-get -yy -q install --no-install-recommends --no-install-suggests --fix-missing \
      dpkg-dev \
      build-essential \
      debhelper \
      dh-systemd \
    && apt-get upgrade -y \
    && apt-get autoremove -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ADD . /src
ADD vendor /go/src

WORKDIR /src

ENTRYPOINT ["go", "run", "/src/build.go"]
