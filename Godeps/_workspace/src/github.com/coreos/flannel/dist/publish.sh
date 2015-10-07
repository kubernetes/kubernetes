#!/bin/bash

if [ $# -lt 1 ]; then
	echo "Usage: $0 tag" >/dev/stderr
	exit 1
fi

TAG=$1

docker push quay.io/coreos/flannel:$TAG
