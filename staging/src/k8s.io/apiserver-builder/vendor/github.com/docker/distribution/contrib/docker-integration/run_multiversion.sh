#!/usr/bin/env bash

# Run the integration tests with multiple versions of the Docker engine

set -e
set -x

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)


if [ "$TMPDIR" != "" ] && [ ! -d "$TMPDIR" ]; then
	mkdir -p $TMPDIR
fi

cachedir=`mktemp -t -d golem-cache.XXXXXX`
trap "rm -rf $cachedir" EXIT

if [ "$1" == "-d" ]; then
       # Drivers to use for Docker engines the tests are going to create.
       STORAGE_DRIVER=${STORAGE_DRIVER:-overlay}

       docker daemon --log-level=panic --storage-driver="$STORAGE_DRIVER" &
       DOCKER_PID=$!

       # Wait for it to become reachable.
       tries=10
       until docker version &> /dev/null; do
               (( tries-- ))
               if [ $tries -le 0 ]; then
                       echo >&2 "error: daemon failed to start"
                       exit 1
               fi
               sleep 1
       done

       trap "kill $DOCKER_PID" EXIT
fi

distimage=$(docker build -q $DIR/../..)
fullversion=$(git describe --match 'v[0-9]*' --dirty='.m' --always)
distversion=${fullversion:1}

echo "Testing image $distimage with distribution version $distversion"

# Pull needed images before invoking golem to get pull time
# These images are defined in golem.conf
time docker pull nginx:1.9
time docker pull golang:1.6
time docker pull registry:0.9.1
time docker pull dmcgowan/token-server:simple
time docker pull dmcgowan/token-server:oauth
time docker pull distribution/golem-runner:0.1-bats

time docker pull docker:1.9.1-dind
time docker pull docker:1.10.3-dind
time docker pull dockerswarm/dind:1.11.0-rc2

golem -cache $cachedir \
	-i "golem-distribution:latest,$distimage,$distversion" \
	-i "golem-dind:latest,docker:1.9.1-dind,1.9.1" \
	-i "golem-dind:latest,docker:1.10.3-dind,1.10.3" \
	-i "golem-dind:latest,dockerswarm/dind:1.11.0-rc2,1.11.0" \
	$DIR

