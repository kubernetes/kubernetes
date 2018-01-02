#!/bin/sh

set -e

errexit() {
	echo "$1"
	exit 1
}

[ "$(uname -s)" == "Darwin" ] || errexit "This script can only be used on a Mac"

[ $# -eq 1 ] || errexit "Usage: $0 install|undo"

BUNDLE="bundles/$(cat VERSION)"
BUNDLE_PATH="$PWD/$BUNDLE"
CLIENT_PATH="$BUNDLE_PATH/cross/darwin/amd64/docker"
DATABASE="$HOME/Library/Containers/com.docker.docker/Data/database"
DATABASE_KEY="$DATABASE/com.docker.driver.amd64-linux/bundle"

[ -d "$DATABASE" ] || errexit "Docker for Mac must be installed for this script"

case "$1" in
"install")
	[ -d "$BUNDLE" ] || errexit "cannot find bundle $BUNDLE"
	[ -e "$CLIENT_PATH" ] || errexit "you need to run make cross first"
	[ -e "$BUNDLE/binary-daemon/dockerd" ] || errexit "you need to build binaries first"
	[ -f "$BUNDLE/binary-client/docker" ] || errexit "you need to build binaries first"
	git -C "$DATABASE" reset --hard >/dev/null
	echo "$BUNDLE_PATH" > "$DATABASE_KEY"
	git -C "$DATABASE" add "$DATABASE_KEY"
	git -C "$DATABASE" commit -m "update bundle to $BUNDLE_PATH"
	rm -f /usr/local/bin/docker
	cp "$CLIENT_PATH" /usr/local/bin
	echo "Bundle installed. Restart Docker to use. To uninstall, reset Docker to factory defaults."
	;;
"undo")
	git -C "$DATABASE" reset --hard >/dev/null
	[ -f "$DATABASE_KEY" ] || errexit "bundle not set"
	git -C "$DATABASE" rm "$DATABASE_KEY"
	git -C "$DATABASE" commit -m "remove bundle"
	rm -f /usr/local/bin/docker
	ln -s "$HOME/Library/Group Containers/group.com.docker/bin/docker" /usr/local/bin
	echo "Bundle removed. Using dev versions may cause issues, a reset to factory defaults is recommended."
	;;
esac
