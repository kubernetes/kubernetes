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

# This script is for master and node instance health monitoring, which is
# packed in kube-manifest tarball. It is executed through a systemd service
# in cluster/gce/gci/<master/node>.yaml. The env variables come from an env
# file provided by the systemd service.

set -o nounset
set -o pipefail

# We simply kill the process when there is a failure. Another systemd service will
# automatically restart the process.
function container_runtime_monitoring {
  local -r max_attempts=5
  local attempt=1
  local -r crictl="${KUBE_HOME}/bin/crictl"
  local -r container_runtime_name="${CONTAINER_RUNTIME_NAME:-docker}"
  # We still need to use `docker ps` when container runtime is "docker". This is because
  # dockershim is still part of kubelet today. When kubelet is down, crictl pods
  # will also fail, and docker will be killed. This is undesirable especially when
  # docker live restore is disabled.
  local healthcheck_command="docker ps"
  if [[ "${CONTAINER_RUNTIME:-docker}" != "docker" ]]; then
    healthcheck_command="${crictl} pods"
  fi
  # Container runtime startup takes time. Make initial attempts before starting
  # killing the container runtime.
  until timeout 60 ${healthcheck_command} > /dev/null; do
    if (( attempt == max_attempts )); then
      echo "Max attempt ${max_attempts} reached! Proceeding to monitor container runtime healthiness."
      break
    fi
    echo "$attempt initial attempt \"${healthcheck_command}\"! Trying again in $attempt seconds..."
    sleep "$(( 2 ** attempt++ ))"
  done
  while true; do
    if ! timeout 60 ${healthcheck_command} > /dev/null; then
      echo "Container runtime ${container_runtime_name} failed!"
      if [[ "$container_runtime_name" == "docker" ]]; then
          # Dump stack of docker daemon for investigation.
          # Log fle name looks like goroutine-stacks-TIMESTAMP and will be saved to
          # the exec root directory, which is /var/run/docker/ on Ubuntu and COS.
          pkill -SIGUSR1 dockerd
      fi
      systemctl kill --kill-who=main "${container_runtime_name}"
      # Wait for a while, as we don't want to kill it again before it is really up.
      sleep 120
    else
      sleep "${SLEEP_SECONDS}"
    fi
  done
}

function kubelet_monitoring {
  echo "Wait for 2 minutes for kubelet to be functional"
  # TODO(andyzheng0831): replace it with a more reliable method if possible.
  sleep 120
  local -r max_seconds=10
  local output=""
  while [ 1 ]; do
    if ! output=$(curl -m "${max_seconds}" -f -s -S http://127.0.0.1:10255/healthz 2>&1); then
      # Print the response and/or errors.
      echo $output
      echo "Kubelet is unhealthy!"
      systemctl kill kubelet
      # Wait for a while, as we don't want to kill it again before it is really up.
      sleep 60
    else
      sleep "${SLEEP_SECONDS}"
    fi
  done
}


############## Main Function ################
if [[ "$#" -ne 1 ]]; then
  echo "Usage: health-monitor.sh <container-runtime/kubelet>"
  exit 1
fi

KUBE_HOME="/home/kubernetes"
KUBE_ENV="${KUBE_HOME}/kube-env"
if [[ ! -e "${KUBE_ENV}" ]]; then
  echo "The ${KUBE_ENV} file does not exist!! Terminate health monitoring"
  exit 1
fi

SLEEP_SECONDS=10
component=$1
echo "Start kubernetes health monitoring for ${component}"
source "${KUBE_ENV}"
if [[ "${component}" == "container-runtime" ]]; then
  container_runtime_monitoring
elif [[ "${component}" == "kubelet" ]]; then
  kubelet_monitoring
else
  echo "Health monitoring for component "${component}" is not supported!"
fi
