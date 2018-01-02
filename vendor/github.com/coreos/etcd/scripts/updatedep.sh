#!/usr/bin/env bash

# A script for updating godep dependencies for the vendored directory /cmd/
# without pulling in etcd itself as a dependency.
#
# update depedency
# 1. edit glide.yaml with version, git SHA
# 2. run ./scripts/updatedep.sh
# 3. it automatically detects new git SHA, and vendors updates to cmd/vendor directory
#
# add depedency
# 1. run ./scripts/updatedep.sh github.com/USER/PROJECT#^1.0.0
#        OR
#        ./scripts/updatedep.sh github.com/USER/PROJECT#9b772b54b3bf0be1eec083c9669766a56332559a
# 2. make sure glide.yaml and glide.lock are updated

if ! [[ "$0" =~ "scripts/updatedep.sh" ]]; then
	echo "must be run from repository root"
	exit 255
fi

rm -rf vendor
mv cmd/vendor vendor

# TODO: glide doesn't play well with symlink
echo "manually deleting etcd-repo symlink in vendor"
rm -f vendor/github.com/coreos/etcd

GLIDE_ROOT="$GOPATH/src/github.com/Masterminds/glide"
GLIDE_SHA=21ff6d397ccca910873d8eaabab6a941c364cc70
go get -d -u github.com/Masterminds/glide
pushd "${GLIDE_ROOT}"
	git reset --hard ${GLIDE_SHA}
	go install
popd

GLIDE_VC_ROOT="$GOPATH/src/github.com/sgotti/glide-vc"
GLIDE_VC_SHA=d96375d23c85287e80296cdf48f9d21c227fa40a
go get -d -u github.com/sgotti/glide-vc
pushd "${GLIDE_VC_ROOT}"
	git reset --hard ${GLIDE_VC_SHA}
	go install
popd

if [ -n "$1" ]; then
	echo "glide get on $(echo $1)"
	matches=`grep "name: $1" glide.lock`
	if [ ! -z "$matches" ]; then
		echo "glide update on $1"
		glide update --strip-vendor $1
	else
		echo "glide get on $1"
		glide get --strip-vendor $1
	fi
else
	echo "glide update on *"
	glide update --strip-vendor
fi;

# TODO: workaround to keep 'github.com/stretchr/testify/assert' in v2 tests
# TODO: remove this after dropping v2
echo "copying github.com/stretchr/testify/assert"
cp -rf vendor/github.com/stretchr/testify/assert ./temp-assert

echo "removing test files"
glide vc --only-code --no-tests

# TODO: remove this after dropping v2
mkdir -p vendor/github.com/stretchr/testify
mv ./temp-assert vendor/github.com/stretchr/testify/assert

mv vendor cmd/

echo "recreating symlink to etcd"
ln -s ../../../../ cmd/vendor/github.com/coreos/etcd

echo "done"

