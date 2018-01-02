#!/bin/bash

IMAGE=${1-heapster:canary}

set -e

pushd $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

godep go build -o heapster -a k8s.io/heapster/metrics
godep go build -o eventer -a k8s.io/heapster/events

docker build -t $IMAGE .
popd
