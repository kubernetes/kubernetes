#!/bin/bash
set -e

# explicit list of os/arch combos that support being a daemon
declare -A daemonSupporting
daemonSupporting=(
	[linux/amd64]=1
)

# if we have our linux/amd64 version compiled, let's symlink it in
if [ -x "$DEST/../binary/docker-$VERSION" ]; then
	mkdir -p "$DEST/linux/amd64"
	(
		cd "$DEST/linux/amd64"
		ln -s ../../../binary/* ./
	)
	echo "Created symlinks:" "$DEST/linux/amd64/"*
fi

for platform in $DOCKER_CROSSPLATFORMS; do
	(
		export DEST="$DEST/$platform" # bundles/VERSION/cross/GOOS/GOARCH/docker-VERSION
		mkdir -p "$DEST"
		ABS_DEST="$(cd "$DEST" && pwd -P)"
		export GOOS=${platform%/*}
		export GOARCH=${platform##*/}
		if [ -z "${daemonSupporting[$platform]}" ]; then
			export LDFLAGS_STATIC_DOCKER="" # we just need a simple client for these platforms
			export BUILDFLAGS=( "${ORIG_BUILDFLAGS[@]/ daemon/}" ) # remove the "daemon" build tag from platforms that aren't supported
		fi
		source "${MAKEDIR}/binary"
	)
done
