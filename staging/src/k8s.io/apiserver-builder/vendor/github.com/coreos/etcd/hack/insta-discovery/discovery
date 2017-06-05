#!/bin/sh

rm -rf infra*.etcd
disc=$(curl -s https://discovery.etcd.io/new?size=3)
echo ETCD_DISCOVERY=${disc} > .env
echo "setup discovery start your cluster"
cat .env
goreman start
