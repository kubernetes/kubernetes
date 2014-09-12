#!/bin/sh
# Usage: ./push.sh [TAG]

docker push gurpartap/guestbook-example:${1:-latest}
