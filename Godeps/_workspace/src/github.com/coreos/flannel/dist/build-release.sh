#!/bin/bash -e

function usage {
	echo "Usage: $0 <version>"
	exit 1
}

function package_tarball {
	builddir="flannel-$VER"
	tarball="flannel-${VER}-${GOOS}-${GOARCH}.tar.gz"

	mkdir -p $builddir
	cp ../bin/flanneld ./mk-docker-opts.sh ../README.md $builddir

	tar cvvfz $tarball "flannel-$VER"
}

VER="$1"
GOARCH="amd64"
GOOS="linux"

if [ "$VER" == "" ]; then
	usage
fi

cur_branch=$(git rev-parse --abbrev-ref HEAD)
git checkout v$VER

./build-docker.sh $VER
./build-aci.sh $VER
package_tarball

# restore the branch
git checkout $cur_branch
