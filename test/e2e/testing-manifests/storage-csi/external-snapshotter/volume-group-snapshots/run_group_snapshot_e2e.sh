#!/bin/sh
# Copyright 2024 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# hack script for running a kind e2e
# must be run with a kubernetes checkout in $PWD (IE from the checkout)
# Usage: SKIP="ginkgo skip regex" FOCUS="ginkgo focus regex" kind-e2e.sh

set -o errexit -o nounset -o xtrace

# Settings:
# SKIP: ginkgo skip regex
# FOCUS: ginkgo focus regex
# LABEL_FILTER: ginkgo label query for selecting tests (see "Spec Labels" in https://onsi.github.io/ginkgo/#filtering-specs)
#
# The default is to focus on conformance tests. Serial tests get skipped when
# parallel testing is enabled. Using LABEL_FILTER instead of combining SKIP and
# FOCUS is recommended (more expressive, easier to read than regexp).
#
# FEATURE_GATES:
#          JSON or YAML encoding of a string/bool map: {"FeatureGateA": true, "FeatureGateB": false}
#          Enables or disables feature gates in the entire cluster.
#          Cannot be used when GA_ONLY=true.
# RUNTIME_CONFIG:
#          JSON or YAML encoding of a string/string (!) map: {"apia.example.com/v1alpha1": "true", "apib.example.com/v1beta1": "false"}
#          Enables API groups in the apiserver via --runtime-config.
#          Cannot be used when GA_ONLY=true.

# cleanup logic for cleanup on exit
CLEANED_UP=false
cleanup() {
  if [ "$CLEANED_UP" = "true" ]; then
    return
  fi
  # KIND_CREATE_ATTEMPTED is true once we: kind create
  if [ "${KIND_CREATE_ATTEMPTED:-}" = true ]; then
    kind "export" logs "${ARTIFACTS}" || true
    kind delete cluster || true
  fi
  rm -f _output/bin/e2e.test || true
  # remove our tempdir, this needs to be last, or it will prevent kind delete
  if [ -n "${TMP_DIR:-}" ]; then
    rm -rf "${TMP_DIR:?}"
  fi
  CLEANED_UP=true
}

# setup signal handlers
# shellcheck disable=SC2317 # this is not unreachable code
signal_handler() {
  if [ -n "${GINKGO_PID:-}" ]; then
    kill -TERM "$GINKGO_PID" || true
  fi
  cleanup
}
trap signal_handler INT TERM

# build kubernetes / node image, e2e binaries
build() {
  # build the node image w/ kubernetes
  kind build node-image -v 1
  # Ginkgo v1 is used by Kubernetes 1.24 and earlier, fallback if v2 is not available.
  GINKGO_SRC_DIR="vendor/github.com/onsi/ginkgo/v2/ginkgo"
  if [ ! -d "$GINKGO_SRC_DIR" ]; then
      GINKGO_SRC_DIR="vendor/github.com/onsi/ginkgo/ginkgo"
  fi
  # make sure we have e2e requirements
  make all WHAT="cmd/kubectl test/e2e/e2e.test ${GINKGO_SRC_DIR}"

  # Ensure the built kubectl is used instead of system
  export PATH="${PWD}/_output/bin:$PATH"
}

# up a cluster with kind
create_cluster() {
  # Default Log level for all components in test clusters
  KIND_CLUSTER_LOG_LEVEL=${KIND_CLUSTER_LOG_LEVEL:-4}

  # JSON or YAML map injected into featureGates config
  feature_gates="${FEATURE_GATES:-{\}}"
  # --runtime-config argument value passed to the API server, again as a map
  runtime_config="${RUNTIME_CONFIG:-{\}}"

  # create the config file
  cat <<EOF > "${ARTIFACTS}/kind-config.yaml"
# config for 1 control plane node and 2 workers (necessary for conformance)
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
networking:
  ipFamily: ${IP_FAMILY:-ipv4}
  kubeProxyMode: ${KUBE_PROXY_MODE:-iptables}
  # don't pass through host search paths
  # TODO: possibly a reasonable default in the future for kind ...
  dnsSearch: []
nodes:
- role: control-plane
- role: worker
- role: worker
featureGates: ${feature_gates}
runtimeConfig: ${runtime_config}
kubeadmConfigPatches:
# v1beta4 for the future (v1.35.0+ ?)
# https://github.com/kubernetes-sigs/kind/issues/3847
# TODO: drop v1beta3 when kind makes the switch
- |
  kind: ClusterConfiguration
  apiVersion: kubeadm.k8s.io/v1beta4
  metadata:
    name: config
  apiServer:
    extraArgs:
      - name: "v"
        value: "${KIND_CLUSTER_LOG_LEVEL}"
  controllerManager:
    extraArgs:
      - name: "v"
        value: "${KIND_CLUSTER_LOG_LEVEL}"
  scheduler:
    extraArgs:
      - name: "v"
        value: "${KIND_CLUSTER_LOG_LEVEL}"
  ---
  kind: InitConfiguration
  apiVersion: kubeadm.k8s.io/v1beta4
  nodeRegistration:
    kubeletExtraArgs:
      - name: "v"
        value: "${KIND_CLUSTER_LOG_LEVEL}"
  ---
  kind: JoinConfiguration
  apiVersion: kubeadm.k8s.io/v1beta4
  nodeRegistration:
    kubeletExtraArgs:
      - name: "v"
        value: "${KIND_CLUSTER_LOG_LEVEL}"
# v1beta3 for v1.23.0 ... ?
- |
  kind: ClusterConfiguration
  apiVersion: kubeadm.k8s.io/v1beta3
  metadata:
    name: config
  apiServer:
    extraArgs:
      "v": "${KIND_CLUSTER_LOG_LEVEL}"
  controllerManager:
    extraArgs:
      "v": "${KIND_CLUSTER_LOG_LEVEL}"
  scheduler:
    extraArgs:
      "v": "${KIND_CLUSTER_LOG_LEVEL}"
  ---
  kind: InitConfiguration
  apiVersion: kubeadm.k8s.io/v1beta3
  nodeRegistration:
    kubeletExtraArgs:
     "v": "${KIND_CLUSTER_LOG_LEVEL}"
  ---
  kind: JoinConfiguration
  apiVersion: kubeadm.k8s.io/v1beta3
  nodeRegistration:
    kubeletExtraArgs:
      "v": "${KIND_CLUSTER_LOG_LEVEL}"
EOF
  # NOTE: must match the number of workers above
  NUM_NODES=2
  # actually create the cluster
  # TODO(BenTheElder): settle on verbosity for this script
  KIND_CREATE_ATTEMPTED=true
  kind create cluster \
    --image=kindest/node:latest \
    --retain \
    --wait=1m \
    -v=3 \
    "--config=${ARTIFACTS}/kind-config.yaml"

  # debug cluster version
  kubectl version

  # Patch kube-proxy to set the verbosity level
  kubectl patch -n kube-system daemonset/kube-proxy \
    --type='json' -p='[{"op": "add", "path": "/spec/template/spec/containers/0/command/-", "value": "--v='"${KIND_CLUSTER_LOG_LEVEL}"'" }]'
}

# run e2es with ginkgo-e2e.sh
run_tests() {
  # IPv6 clusters need some CoreDNS changes in order to work in k8s CI:
  # 1. k8s CI doesn´t offer IPv6 connectivity, so CoreDNS should be configured
  # to work in an offline environment:
  # https://github.com/coredns/coredns/issues/2494#issuecomment-457215452
  # 2. k8s CI adds following domains to resolv.conf search field:
  # c.k8s-prow-builds.internal google.internal.
  # CoreDNS should handle those domains and answer with NXDOMAIN instead of SERVFAIL
  # otherwise pods stops trying to resolve the domain.
  if [ "${IP_FAMILY:-ipv4}" = "ipv6" ]; then
    # Get the current config
    original_coredns=$(kubectl get -oyaml -n=kube-system configmap/coredns)
    echo "Original CoreDNS config:"
    echo "${original_coredns}"
    # Patch it
    fixed_coredns=$(
      printf '%s' "${original_coredns}" | sed \
        -e 's/^.*kubernetes cluster\.local/& internal/' \
        -e '/^.*upstream$/d' \
        -e '/^.*fallthrough.*$/d' \
        -e '/^.*forward . \/etc\/resolv.conf$/d' \
        -e '/^.*loop$/d' \
    )
    echo "Patched CoreDNS config:"
    echo "${fixed_coredns}"
    printf '%s' "${fixed_coredns}" | kubectl apply -f -
  fi

  # ginkgo regexes and label filter
  SKIP="${SKIP:-}"
  FOCUS="${FOCUS:-}"
  LABEL_FILTER="${LABEL_FILTER:-}"
  if [ -z "${FOCUS}" ] && [ -z "${LABEL_FILTER}" ]; then
    FOCUS="\\[Conformance\\]"
  fi
  # if we set PARALLEL=true, skip serial tests set --ginkgo-parallel
  if [ "${PARALLEL:-false}" = "true" ]; then
    export GINKGO_PARALLEL=y
    if [ -z "${SKIP}" ]; then
      SKIP="\\[Serial\\]"
    else
      SKIP="\\[Serial\\]|${SKIP}"
    fi
  fi

  # setting this env prevents ginkgo e2e from trying to run provider setup
  export KUBERNETES_CONFORMANCE_TEST='y'
  # setting these is required to make RuntimeClass tests work ... :/
  export KUBE_CONTAINER_RUNTIME=remote
  export KUBE_CONTAINER_RUNTIME_ENDPOINT=unix:///run/containerd/containerd.sock
  export KUBE_CONTAINER_RUNTIME_NAME=containerd
  export SNAPSHOTTER_VERSION="${SNAPSHOTTER_VERSION:-v8.4.0}"
  echo "SNAPSHOTTER_VERSION is $SNAPSHOTTER_VERSION"

  # Enable VolumeGroupSnapshot tests in csi-driver-hostpath
  export CSI_PROW_ENABLE_GROUP_SNAPSHOT=true

  kubectl apply -f ./cluster/addons/volumesnapshots/crd/snapshot.storage.k8s.io_volumesnapshotclasses.yaml || exit 1
  kubectl apply -f ./cluster/addons/volumesnapshots/crd/snapshot.storage.k8s.io_volumesnapshotcontents.yaml || exit 1
  kubectl apply -f ./cluster/addons/volumesnapshots/crd/snapshot.storage.k8s.io_volumesnapshots.yaml || exit 1
  kubectl apply -f test/e2e/testing-manifests/storage-csi/external-snapshotter/groupsnapshot.storage.k8s.io_volumegroupsnapshotclasses.yaml || exit 1
  kubectl apply -f test/e2e/testing-manifests/storage-csi/external-snapshotter/groupsnapshot.storage.k8s.io_volumegroupsnapshotcontents.yaml || exit 1
  kubectl apply -f test/e2e/testing-manifests/storage-csi/external-snapshotter/groupsnapshot.storage.k8s.io_volumegroupsnapshots.yaml || exit 1

  kubectl apply -f https://raw.githubusercontent.com/kubernetes-csi/external-snapshotter/refs/tags/"${SNAPSHOTTER_VERSION}"/deploy/kubernetes/snapshot-controller/rbac-snapshot-controller.yaml || exit 1
  curl -s https://raw.githubusercontent.com/kubernetes-csi/external-snapshotter/refs/tags/"${SNAPSHOTTER_VERSION}"/deploy/kubernetes/snapshot-controller/setup-snapshot-controller.yaml | \
awk '/--leader-election=true/ {print; print "            - \"--feature-gates=CSIVolumeGroupSnapshot=true\""; next}1' |  \
sed "s|image: registry.k8s.io/sig-storage/snapshot-controller:.*|image: registry.k8s.io/sig-storage/snapshot-controller:${SNAPSHOTTER_VERSION}|" | \
 kubectl apply -f - || exit 1


  # ginkgo can take forever to exit, so we run it in the background and save the
  # PID, bash will not run traps while waiting on a process, but it will while
  # running a builtin like `wait`, saving the PID also allows us to forward the
  # interrupt
  ./hack/ginkgo-e2e.sh \
    '--provider=skeleton' "--num-nodes=${NUM_NODES}" \
    "--ginkgo.focus=${FOCUS}" "--ginkgo.skip=${SKIP}" "--ginkgo.label-filter=${LABEL_FILTER}" \
    "--report-dir=${ARTIFACTS}" '--disable-log-dump=true' &
  GINKGO_PID=$!
  wait "$GINKGO_PID"
}

main() {
  # create temp dir and setup cleanup
  TMP_DIR=$(mktemp -d)

  # ensure artifacts (results) directory exists when not in CI
  export ARTIFACTS="${ARTIFACTS:-${PWD}/_artifacts}"
  mkdir -p "${ARTIFACTS}"

  # export the KUBECONFIG to a unique path for testing
  KUBECONFIG="${HOME}/.kube/kind-test-config"
  export KUBECONFIG
  echo "exported KUBECONFIG=${KUBECONFIG}"

  # debug kind version
  kind version
  cp test/e2e/testing-manifests/storage-csi/external-snapshotter/volume-group-snapshots/csi-hostpath-plugin.yaml test/e2e/testing-manifests/storage-csi/hostpath/hostpath/csi-hostpath-plugin.yaml || exit 1
  # build kubernetes
  build
  # in CI attempt to release some memory after building
  if [ -n "${KUBETEST_IN_DOCKER:-}" ]; then
    sync || true
    echo 1 > /proc/sys/vm/drop_caches || true
  fi

  
  # create the cluster and run tests
  res=0
  create_cluster || res=$?
  run_tests || res=$?
  cleanup || res=$?
  exit $res
}

main
