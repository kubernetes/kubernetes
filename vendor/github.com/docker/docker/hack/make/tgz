#!/bin/bash

CROSS="$DEST/../cross"

set -e

if [ ! -d "$CROSS/linux/amd64" ]; then
	echo >&2 'error: binary and cross must be run before tgz'
	false
fi

for d in "$CROSS/"*/*; do
	GOARCH="$(basename "$d")"
	GOOS="$(basename "$(dirname "$d")")"
	BINARY_NAME="docker-$VERSION"
	BINARY_EXTENSION="$(export GOOS && binary_extension)"
	BINARY_FULLNAME="$BINARY_NAME$BINARY_EXTENSION"
	mkdir -p "$DEST/$GOOS/$GOARCH"
	TGZ="$DEST/$GOOS/$GOARCH/$BINARY_NAME.tgz"

	mkdir -p "$DEST/build"

	mkdir -p "$DEST/build/usr/local/bin"
	cp -L "$d/$BINARY_FULLNAME" "$DEST/build/usr/local/bin/docker$BINARY_EXTENSION"

	tar --numeric-owner --owner 0 -C "$DEST/build" -czf "$TGZ" usr

	hash_files "$TGZ"

	rm -rf "$DEST/build"

	echo "Created tgz: $TGZ"
done
