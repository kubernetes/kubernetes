#!/usr/bin/env sh

DISTRO=$(cat /etc/os-release | grep "^ID=" | cut -d "=" -f 2)

if [ "$DISTRO" = "alpine" ]; then
  apk update && apk add curl
elif [ "$DISTRO" = "ubuntu" ]; then
  apt-get update -y && \
  apt-get install -y curl && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
fi