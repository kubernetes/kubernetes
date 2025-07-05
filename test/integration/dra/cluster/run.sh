#!/bin/sh

set -ex

killall etcd || true
sudo rm -rf /tmp/ginkgo* /tmp/*.log /var/run/kubernetes /var/run/cdi /var/lib/kubelet/plugins_registry /var/lib/kubelet/plugins /var/lib/kubelet/*_state /var/lib/kubelet/checkpoints /tmp/artifacts
sudo mkdir /var/lib/kubelet/plugins_registry
sudo mkdir /var/lib/kubelet/plugins
sudo mkdir /var/run/cdi
sudo chown $(id -u) /var/lib/kubelet/plugins_registry /var/lib/kubelet/plugins /var/run/cdi
export ARTIFACTS=/tmp/artifacts
export KUBERNETES_SERVER_BIN_DIR="$(pwd)/_output/local/bin/$(go env GOOS)/$(go env GOARCH)"
export KUBERNETES_SERVER_CACHE_DIR="${KUBERNETES_SERVER_BIN_DIR}/cache-dir"

exec "$@"
