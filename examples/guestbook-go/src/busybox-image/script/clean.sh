#!/bin/sh
# Usage: ./clean.sh

docker rm -f guestbook-example-build 2> /dev/null
docker rmi -f gurpartap/guestbook-example-build
docker rmi -f gurpartap/guestbook-example
