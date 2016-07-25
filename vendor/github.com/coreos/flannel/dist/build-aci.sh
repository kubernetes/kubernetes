#!/bin/bash
set -e

if [ $# -ne 1 ]; then
	echo "Usage: $0 tag" >/dev/stderr
	exit 1
fi

tag=$1

tgt=$(mktemp -d)

./build-docker.sh $tag

docker save -o ${tgt}/flannel-${tag}.docker quay.io/coreos/flannel:${tag}
docker2aci ${tgt}/flannel-${tag}.docker

VOL1=run-flannel,path=/run/flannel,readOnly=false
VOL2=etc-ssl-etcd,path=/etc/ssl/etcd,readOnly=true
VOL3=dev-net,path=/dev/net,readOnly=false
actool patch-manifest --replace --capability=CAP_NET_ADMIN --mounts=${VOL1}:${VOL2}:${VOL3} quay.io-coreos-flannel-${tag}.aci

# Cleanup
rm -rf $tgt
