#!/usr/bin/env sh

DISTRO=$(cat /etc/os-release | grep "^ID=" | cut -d "=" -f 2)

# install necessary packages:
# - curl, nc: used by a lot of e2e tests
# - iproute2: includes ss used in NodePort tests
if [ "$DISTRO" = "alpine" ]; then
  apk --update add curl netcat-openbsd iproute2 && rm -rf /var/cache/apk/*
elif [ "$DISTRO" = "ubuntu" ]; then
  apt-get update -y && \
  apt-get install -y curl netcat-openbsd iproute2 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
fi