#!/bin/bash -e

# Generate vSphere Cloud Provider config from govc env and run local-up-cluster.sh
# Assumes the create.sh NFS configuration has been applied.

GOVC_NETWORK=${GOVC_NETWORK:-"VM Network"}
GOVC_DATACENTER=${GOVC_DATACENTER:-"$(govc find / -type d)"}
GOVC_DATACENTER="$(basename "$GOVC_DATACENTER")"
GOVC_CLUSTER=${GOVC_CLUSTER:-"$(govc find / -type c -type r)"}

oneline() {
  awk '{printf "%s\\n", $0}' "$1" # make gcfg happy
}

username="$(govc env GOVC_USERNAME)"
password="$(govc env GOVC_PASSWORD)"
if [ -n "$GOVC_CERTIFICATE" ] ; then
  username="$(oneline "$GOVC_CERTIFICATE")"
  password="$(oneline "$GOVC_PRIVATE_KEY")"
fi

cat <<EOF | tee vcp.conf
[Global]
        insecure-flag = "$(govc env GOVC_INSECURE)"

[VirtualCenter "$(govc env -x GOVC_URL_HOST)"]
        user = "$username"
        password = "$password"
        port = "$(govc env -x GOVC_URL_PORT)"
        datacenters = "$(basename "$GOVC_DATACENTER")"

[Workspace]
        server = "$(govc env -x GOVC_URL_HOST)"
        datacenter = "$GOVC_DATACENTER"
        folder = "vm"
        default-datastore = "$GOVC_DATACENTER"
        resourcepool-path = "$GOVC_CLUSTER/Resources"
[Disk]
        scsicontrollertype = pvscsi

[Network]
        public-network = "$GOVC_NETWORK"
EOF

k8s="$GOPATH/src/k8s.io/kubernetes"
make -C "$k8s" WHAT="cmd/kubectl cmd/hyperkube"

ip=$(govc vm.ip -a -v4 "$USER-ubuntu-16.04")

# shellcheck disable=2029
ssh -tt </dev/null -i ~/.vagrant.d/insecure_private_key -L 8080:127.0.0.1:8080 "vagrant@$ip" \
    CLOUD_PROVIDER=vsphere CLOUD_CONFIG="$PWD/vcp.conf" LOG_DIR="$LOG_DIR" \
    PATH="$PATH:$k8s/third_party/etcd" "$k8s/hack/local-up-cluster.sh" -O
