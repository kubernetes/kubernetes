#!/bin/bash

# Run a docker container with network namespace set up by the
# CNI plugins.

# Example usage: ./docker-run.sh --rm busybox /sbin/ifconfig

contid=$(docker run -d --net=none busybox:latest /bin/sleep 10000000)
pid=$(docker inspect -f '{{ .State.Pid }}' $contid)
netnspath=/proc/$pid/ns/net

./exec-plugins.sh add $contid $netnspath

function cleanup() {
	./exec-plugins.sh del $contid $netnspath
	docker rm -f $contid >/dev/null
}
trap cleanup EXIT

docker run --net=container:$contid $@

