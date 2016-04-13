#!/bin/bash

# This script signs the deliverables from release-deb and release-rpm
# with a designated GPG key.

: ${DOCKER_RELEASE_DIR:=$DEST}
APTDIR=$DOCKER_RELEASE_DIR/apt/repo
YUMDIR=$DOCKER_RELEASE_DIR/yum/repo

if [ -z "$GPG_PASSPHRASE" ]; then
	echo >&2 'you need to set GPG_PASSPHRASE in order to sign artifacts'
	exit 1
fi

if [ ! -d $APTDIR ] && [ ! -d $YUMDIR ]; then
	echo >&2 'release-rpm or release-deb must be run before sign-repos'
	exit 1
fi

sign_packages(){
	# sign apt repo metadata
	if [ -d $APTDIR ]; then
		# create file with public key
		gpg --armor --export releasedocker > "$DOCKER_RELEASE_DIR/apt/gpg"

		# sign the repo metadata
		for F in $(find $APTDIR -name Release); do
			gpg -u releasedocker --passphrase "$GPG_PASSPHRASE" \
				--armor --sign --detach-sign \
				--batch --yes \
				--output "$F.gpg" "$F"
		done
	fi

	# sign yum repo metadata
	if [ -d $YUMDIR ]; then
		# create file with public key
		gpg --armor --export releasedocker > "$DOCKER_RELEASE_DIR/yum/gpg"

		# sign the repo metadata
		for F in $(find $YUMDIR -name repomd.xml ); do
			gpg -u releasedocker --passphrase "$GPG_PASSPHRASE" \
				--armor --sign --detach-sign \
				--batch --yes \
				--output "$F.asc" "$F"
		done
	fi
}

sign_packages
