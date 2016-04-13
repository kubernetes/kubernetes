#!/bin/bash

IMAGE=${1-heapster:canary}

set -e

pushd $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

godep go build -a k8s.io/heapster

docker build -t $IMAGE .
popd
