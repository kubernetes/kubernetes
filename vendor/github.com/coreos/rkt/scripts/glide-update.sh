#!/usr/bin/env bash
#
# Update vendored dedendencies.
#
set -e

if ! [[ "$PWD" = "$GOPATH/src/github.com/coreos/rkt" ]]; then
  echo "must be run from \$GOPATH/src/github.com/coreos/rkt"
  exit 255
fi

if [ ! $(command -v glide) ]; then
	echo "glide: command not found"
	exit 255
fi

if [ ! $(command -v glide-vc) ]; then
	echo "glide-vc: command not found"
	exit 255
fi

glide update --strip-vendor
glide-vc --only-code --no-tests --keep="**/*.json.in" --use-lock-file
