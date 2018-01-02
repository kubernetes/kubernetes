#! /bin/bash

pushd $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
docker build -t heapster_kafka:canary .
popd
