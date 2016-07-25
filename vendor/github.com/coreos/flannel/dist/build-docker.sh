#!/bin/bash
set -e

if [ $# -ne 1 ]; then
	echo "Usage: $0 tag" >/dev/stderr
	exit 1
fi

tag=$1

tgt=$(mktemp -d)

# Build flannel inside 
docker run -v `pwd`/../:/opt/flannel -i -t golang:1.4.2 /bin/bash -c "cd /opt/flannel && ./build"

# Generate Dockerfile into target tmp dir
cat <<DF >${tgt}/Dockerfile
FROM quay.io/coreos/flannelbox:1.0
MAINTAINER Eugene Yakubovich <eugene.yakubovich@coreos.com>
ADD ./flanneld /opt/bin/
ADD ./mk-docker-opts.sh /opt/bin/
CMD /opt/bin/flanneld
DF

# Copy artifcats into target dir and build the image
cp ../bin/flanneld $tgt
cp ./mk-docker-opts.sh $tgt
docker build -t quay.io/coreos/flannel:${tag} $tgt

# Cleanup
rm -rf $tgt
