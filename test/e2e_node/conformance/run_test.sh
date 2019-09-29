#!/usr/bin/env bash

# Copyright 2016 The Kubernetes Authors.
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

# This script is only for demonstrating how to use the node test container. In
# production environment, kubelet bootstrap will be more complicated, user
# should configure the node test container accordingly.
# In addition, this script will also be used in the node e2e test to let it use
# the containerized test suite.

# TODO(random-liu): Use standard installer to install kubelet.
# TODO(random-liu): Use standard tool to start kubelet in production way (such
# as systemd, supervisord etc.)

# Refresh sudo credentials if needed
if ping -c 1 -q metadata.google.internal &> /dev/null; then
  echo 'Running on GCE, not asking for sudo credentials'
elif sudo --non-interactive "$(which bash)" -c true 2> /dev/null; then
  # if we can run bash without a password, it's a pretty safe bet that either
  # we can run any command without a password, or that sudo credentials
  # are already cached - and they've just been re-cached
  echo 'No need to refresh sudo credentials'
else
  echo 'Updating sudo credentials'
  sudo -v || exit 1
fi

# FOCUS is ginkgo focus to select which tests to run. By default, FOCUS is
# initialized as "\[Conformance\]" in the test container to run all conformance
# test.
FOCUS=${FOCUS:-""}

# SKIP is ginkgo skip to select which tests to skip. By default, SKIP is
# initialized as "\[Flaky\]|\[Serial\]" in the test container skipping all
# flaky and serial test.
SKIP=${SKIP:-""}

# TEST_ARGS is the test arguments. It could be used to override default test
# arguments in the container.
TEST_ARGS=${TEST_ARGS:-""}

# REGISTRY is the image registry for node test image.
REGISTRY=${REGISTRY:-"k8s.gcr.io"}

# ARCH is the architecture of current machine, the script will use this to
# select corresponding test container image.
ARCH=${ARCH:-"amd64"}

# VERSION is the version of the test container image.
VERSION=${VERSION:-"0.2"}

# KUBELET_BIN is the kubelet binary name. If it is not specified, use the
# default binary name "kubelet".
KUBELET_BIN=${KUBELET_BIN:-"kubelet"}

# KUBELET is the kubelet binary path. If it is not specified, assume kubelet is
# in PATH.
KUBELET=${KUBELET:-"$(which "$KUBELET_BIN")"}

# LOG_DIR is the absolute path of the directory where the test will collect all
# logs to. By default, use the current directory.
LOG_DIR=${LOG_DIR:-$(pwd)}
mkdir -p "$LOG_DIR"

# NETWORK_PLUGIN is the network plugin used by kubelet. Do not use network
# plugin by default.
NETWORK_PLUGIN=${NETWORK_PLUGIN:-""}

# CNI_CONF_DIR is the path to network plugin binaries.
CNI_CONF_DIR=${CNI_CONF_DIR:-""}

# CNI_BIN_DIR is the path to network plugin config files.
CNI_BIN_DIR=${CNI_BIN_DIR:-""}

# KUBELET_KUBECONFIG is the path to a kubeconfig file, specifying how to connect to the API server.
KUBELET_KUBECONFIG=${KUBELET_KUBECONFIG:-"/var/lib/kubelet/kubeconfig"}

# Creates a kubeconfig file for the kubelet.
# Args: address (e.g. "http://localhost:8080"), destination file path
function create-kubelet-kubeconfig() {
  local api_addr="${1}"
  local dest="${2}"
  local dest_dir
  dest_dir="$(dirname "${dest}")"
  mkdir -p "${dest_dir}" &>/dev/null || sudo mkdir -p "${dest_dir}"
  sudo=$(test -w "${dest_dir}" || echo "sudo -E")
  cat <<EOF | ${sudo} tee "${dest}" > /dev/null
apiVersion: v1
kind: Config
clusters:
  - cluster:
      server: ${api_addr}
    name: local
contexts:
  - context:
      cluster: local
    name: local
current-context: local
EOF
}

# start_kubelet starts kubelet and redirect kubelet log to $LOG_DIR/kubelet.log.
kubelet_log=kubelet.log
start_kubelet() {
  echo "Creating kubelet.kubeconfig"
  create-kubelet-kubeconfig "http://localhost:8080" "${KUBELET_KUBECONFIG}"
  echo "Starting kubelet..."
  # we want to run this command as root but log the file to a normal user file
  # (so disable SC2024)
  # shellcheck disable=SC2024
  if ! sudo -b "${KUBELET}" "$@" &>"${LOG_DIR}/${kubelet_log}"; then 
    echo "Failed to start kubelet"
    exit 1
  fi
}

# wait_kubelet retries for 10 times for kubelet to be ready by checking http://127.0.0.1:10255/healthz.
wait_kubelet() {
  echo "Health checking kubelet..."
  healthCheckURL=http://127.0.0.1:10255/healthz
  local maxRetry=10
  local cur=1
  while [ $cur -le $maxRetry ]; do
    if curl -s $healthCheckURL > /dev/null; then
      echo "Kubelet is ready"
      break
    fi
    if [ $cur -eq $maxRetry ]; then
      echo "Health check exceeds max retry"
      exit 1
    fi
    echo "Kubelet is not ready"
    sleep 1
    ((cur++))
  done
}

# kill_kubelet kills kubelet.
kill_kubelet() {
  echo "Stopping kubelet..."
  if ! sudo pkill "${KUBELET_BIN}"; then
    echo "Failed to stop kubelet."
    exit 1
  fi
}

# run_test runs the node test container.
run_test() {
  env=""
  if [ -n "$FOCUS" ]; then
    env="$env -e FOCUS=\"$FOCUS\""
  fi
  if [ -n "$SKIP" ]; then
    env="$env -e SKIP=\"$SKIP\""
  fi
  if [ -n "$TEST_ARGS" ]; then
    env="$env -e TEST_ARGS=\"$TEST_ARGS\""
  fi
  # The test assumes that inside the container:
  # * kubelet manifest path is mounted to the same path;
  # * log collect directory is mounted to /var/result;
  # * root file system is mounted to /rootfs.
  sudo sh -c "docker run -it --rm --privileged=true --net=host -v /:/rootfs \
    -v $config_dir:$config_dir -v $LOG_DIR:/var/result ${env} $REGISTRY/node-test-$ARCH:$VERSION"
}

# Check whether kubelet is running. If kubelet is running, tell the user to stop
# it before running the test.
pid=$(pidof "${KUBELET_BIN}")
if [ -n "$pid" ]; then
  echo "Kubelet is running (pid=$pid), please stop it before running the test."
  exit 1
fi

volume_stats_agg_period=10s
serialize_image_pulls=false
config_dir=$(mktemp -d)
file_check_frequency=10s
pod_cidr=10.100.0.0/24
log_level=4
start_kubelet --kubeconfig "${KUBELET_KUBECONFIG}" \
  --volume-stats-agg-period $volume_stats_agg_period \
  --serialize-image-pulls=$serialize_image_pulls \
  --pod-manifest-path "${config_dir}" \
  --file-check-frequency $file_check_frequency \
  --pod-cidr=$pod_cidr \
  --runtime-cgroups=/docker-daemon \
  --kubelet-cgroups=/kubelet \
  --system-cgroups=/system \
  --cgroup-root=/ \
  "--network-plugin=${NETWORK_PLUGIN}" \
  "--cni-conf-dir=${CNI_CONF_DIR}" \
  "--cni-bin-dir=${CNI_BIN_DIR}" \
  --v=$log_level \
  --logtostderr

wait_kubelet

run_test

kill_kubelet

# Clean up the kubelet config directory
sudo rm -rf "${config_dir}"
