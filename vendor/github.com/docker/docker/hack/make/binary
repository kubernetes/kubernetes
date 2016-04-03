#!/bin/bash
set -e

BINARY_NAME="docker-$VERSION"
BINARY_EXTENSION="$(binary_extension)"
BINARY_FULLNAME="$BINARY_NAME$BINARY_EXTENSION"

source "${MAKEDIR}/.go-autogen"

echo "Building: $DEST/$BINARY_FULLNAME"
go build \
	-o "$DEST/$BINARY_FULLNAME" \
	"${BUILDFLAGS[@]}" \
	-ldflags "
		$LDFLAGS
		$LDFLAGS_STATIC_DOCKER
	" \
	./docker

echo "Created binary: $DEST/$BINARY_FULLNAME"
ln -sf "$BINARY_FULLNAME" "$DEST/docker$BINARY_EXTENSION"

hash_files "$DEST/$BINARY_FULLNAME"
