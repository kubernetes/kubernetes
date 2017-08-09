#!/bin/sh
set -e

# no arguments passed
# or first arg is `-f` or `--some-option`
if [ "$#" -eq 0 -o "${1#-}" != "$1" ]; then
  # add our default arguments
  set -- dockerd \
    --host=unix:///var/run/docker.sock \
    --host=tcp://0.0.0.0:2375 \
    --storage-driver=vfs \
    --bip=10.0.0.1/8 \
    "$@"
fi

if [ "$1" = 'dockerd' ]; then
  # if we're running Docker, let's pipe through dind
  # (and we'll run dind explicitly with "sh" since its shebang is /bin/bash)
  set -- sh "$(which dind)" "$@"
fi

exec "$@"
