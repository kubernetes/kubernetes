#!/bin/bash
set -e

IAMSTATIC="true"
source "${MAKEDIR}/.go-autogen"

# dockerinit still needs to be a static binary, even if docker is dynamic
go build \
	-o "$DEST/dockerinit-$VERSION" \
	"${BUILDFLAGS[@]}" \
	-ldflags "
		$LDFLAGS
		$LDFLAGS_STATIC
		-extldflags \"$EXTLDFLAGS_STATIC\"
	" \
	./dockerinit

echo "Created binary: $DEST/dockerinit-$VERSION"
ln -sf "dockerinit-$VERSION" "$DEST/dockerinit"

sha1sum=
if command -v sha1sum &> /dev/null; then
	sha1sum=sha1sum
elif command -v shasum &> /dev/null; then
	# Mac OS X - why couldn't they just use the same command name and be happy?
	sha1sum=shasum
else
	echo >&2 'error: cannot find sha1sum command or equivalent'
	exit 1
fi

# sha1 our new dockerinit to ensure separate docker and dockerinit always run in a perfect pair compiled for one another
export DOCKER_INITSHA1=$($sha1sum "$DEST/dockerinit-$VERSION" | cut -d' ' -f1)
