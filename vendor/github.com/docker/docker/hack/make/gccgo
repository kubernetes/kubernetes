#!/bin/bash
set -e

BINARY_NAME="docker-$VERSION"
BINARY_EXTENSION="$(binary_extension)"
BINARY_FULLNAME="$BINARY_NAME$BINARY_EXTENSION"

source "${MAKEDIR}/.go-autogen"

if [[ "${BUILDFLAGS[@]}" =~ 'netgo ' ]]; then
	EXTLDFLAGS_STATIC_DOCKER+=' -lnetgo'
fi
go build -compiler=gccgo \
	-o "$DEST/$BINARY_FULLNAME" \
	"${BUILDFLAGS[@]}" \
	-gccgoflags "
		-g
		$EXTLDFLAGS_STATIC_DOCKER
		-Wl,--no-export-dynamic
		-ldl
	" \
	./docker

echo "Created binary: $DEST/$BINARY_FULLNAME"
ln -sf "$BINARY_FULLNAME" "$DEST/docker$BINARY_EXTENSION"

hash_files "$DEST/$BINARY_FULLNAME"
