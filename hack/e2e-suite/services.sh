#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

# Verifies that services and portals work.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..

: ${KUBE_VERSION_ROOT:=${KUBE_ROOT}}
: ${KUBECTL:="${KUBE_VERSION_ROOT}/cluster/kubectl.sh"}
: ${KUBE_CONFIG_FILE:="config-test.sh"}

export KUBECTL KUBE_CONFIG_FILE

source "${KUBE_ROOT}/cluster/kube-env.sh"
source "${KUBE_VERSION_ROOT}/cluster/${KUBERNETES_PROVIDER}/util.sh"

prepare-e2e

function error() {
  echo "$@" >&2
  exit 1
}

function sort_args() {
  printf "%s\n" "$@" | sort -n | tr '\n\r' ' ' | sed 's/  */ /g'
}

# Join args $2... with $1 between them.
# Example: join ", " x y z   =>   x, y, z
function join() {
  local sep item
  sep=$1
  shift
  echo -n "${1:-}"
  shift
  for item; do
    echo -n "${sep}${item}"
  done
  echo
}

svcs_to_clean=()
function do_teardown() {
  local svc
  for svc in "${svcs_to_clean[@]:+${svcs_to_clean[@]}}"; do
    stop_service "${svc}"
  done
}

# Args:
#   $1: service name
#   $2: service port
#   $3: service replica count
#   $4: public IPs (optional, string e.g. "1.2.3.4 5.6.7.8")
function start_service() {
  echo "Starting service '$1' on port $2 with $3 replicas"
  svcs_to_clean+=("$1")
  ${KUBECTL} create -f - << __EOF__
{
  "kind": "ReplicationController",
  "apiVersion": "v1beta3",
  "metadata": {
    "name": "$1",
    "namespace": "default",
    "labels": {
      "name": "$1"
    }
  },
  "spec": {
    "replicas": $3,
    "selector": {
      "name": "$1"
    },
    "template": {
      "metadata": {
        "labels": {
          "name": "$1"
        }
      },
      "spec": {
        "containers": [
          {
            "name": "$1",
            "image": "gcr.io/google_containers/serve_hostname:1.1",
            "ports": [
              {
                "containerPort": 9376,
                "protocol": "TCP"
              }
            ]
          }
        ]
      }
    }
  }
}
__EOF__
  # Convert '1.2.3.4 5.6.7.8' => '"1.2.3.4", "5.6.7.8"'
  local ip ips_array=() public_ips
  for ip in ${4:-}; do
    ips_array+=("\"${ip}\"")
  done
  public_ips=$(join ", " "${ips_array[@]:+${ips_array[@]}}")
  ${KUBECTL} create -f - << __EOF__
{
  "kind": "Service",
  "apiVersion": "v1beta3",
  "metadata": {
    "name": "$1",
    "namespace": "default",
    "labels": {
      "name": "$1"
    }
  },
  "spec": {
    "ports": [
      {
        "protocol": "TCP",
        "port": $2,
        "targetPort": 9376
      }
    ],
    "selector": {
      "name": "$1"
    },
    "publicIPs": [ ${public_ips} ]
  }
}
__EOF__
}

# Args:
#   $1: service name
function stop_service() {
  echo "Stopping service '$1'"
  ${KUBECTL} stop rc "$1" || true
  ${KUBECTL} delete services "$1" || true
}

# Args:
#   $1: service name
#   $2: expected pod count
function query_pods() {
  # This fails very occasionally, so retry a bit.
  local pods_unsorted=()
  local i
  for i in $(seq 1 10); do
    pods_unsorted=($(${KUBECTL} get pods -o template \
        '--template={{range.items}}{{.metadata.name}} {{end}}' \
        '--api-version=v1beta3' \
        -l name="$1"))
    found="${#pods_unsorted[*]}"
    if [[ "${found}" == "$2" ]]; then
      break
    fi
    sleep 3
  done
  if [[ "${found}" != "$2" ]]; then
    error "Failed to query pods for $1: expected $2, found ${found}"
  fi

  # The "return" is a sorted list of pod IDs.
  sort_args "${pods_unsorted[@]}"
}

# Args:
#   $1: service name
#   $2: pod count
function wait_for_pods() {
  echo "Querying pods in $1"
  local pods_sorted=$(query_pods "$1" "$2")
  printf '\t%s\n' ${pods_sorted}

  # Container turn up on a clean cluster can take a while for the docker image
  # pulls.  Wait a generous amount of time.
  # TODO: Sometimes pods change underneath us, which makes the GET fail (404).
  # Maybe this test can be loosened and still be useful?
  pods_needed=$2
  local i
  for i in $(seq 1 30); do
    echo "Waiting for ${pods_needed} pods to become 'running'"
    pods_needed="$2"
    for id in ${pods_sorted}; do
      status=$(${KUBECTL} get pods "${id}" -o template --template='{{.status.phase}}' --api-version=v1beta3)
      if [[ "${status}" == "Running" ]]; then
        pods_needed=$((pods_needed-1))
      fi
    done
    if [[ "${pods_needed}" == 0 ]]; then
      break
    fi
    sleep 3
  done
  if [[ "${pods_needed}" -gt 0 ]]; then
    error "Pods for $1 did not come up in time"
  fi
}

# Args:
#   $1: service name
#   $2: service IP
#   $3: service port
#   $4: pod count
#   $5: pod IDs
function wait_for_service_up() {
  local i
  local found_pods
  for i in $(seq 1 20); do
    results=($(ssh-to-node "${test_node}" "
        set -e;
        for i in $(seq -s' ' 1 $4); do
          curl -s --connect-timeout 1 http://$2:$3;
          echo;
        done | sort | uniq
        "))

    found_pods=$(sort_args "${results[@]:+${results[@]}}")
    echo "Checking if ${found_pods} == ${5}"
    if [[ "${found_pods}" == "$5" ]]; then
      break
    fi
    echo "Waiting for endpoints to propagate"
    sleep 3
  done
  if [[ "${found_pods}" != "$5" ]]; then
    error "Endpoints did not propagate in time"
  fi
}

# Args:
#   $1: service name
#   $2: service IP
#   $3: service port
function wait_for_service_down() {
  local i
  for i in $(seq 1 15); do
    $(ssh-to-node "${test_node}" "
        curl -s --connect-timeout 2 "http://$2:$3" >/dev/null 2>&1 && exit 1 || exit 0;
        ") && break
    echo "Waiting for $1 to go down"
    sleep 2
  done
}

# Args:
#   $1: service name
#   $2: service IP
#   $3: service port
#   $4: pod count
#   $5: pod IDs
function verify_from_container() {
  results=($(ssh-to-node "${test_node}" "
      set -e;
      sudo docker pull busybox >/dev/null;
      sudo docker run busybox sh -c '
          for i in $(seq -s' ' 1 $4); do
            ok=false
            for j in $(seq -s' ' 1 10); do
              if wget -q -T 5 -O - http://$2:$3; then
                echo
                ok=true
                break
              fi
              sleep 1
            done
            if [[ \${ok} == false ]]; then
              exit 1
            fi
          done
      '")) \
      || error "testing $1 portal from container failed"
  found_pods=$(sort_args "${results[@]}")
  if [[ "${found_pods}" != "$5" ]]; then
    error -e "$1 portal failed from container, expected:\n
        $(printf '\t%s\n' $5)\n
        got:\n
        $(printf '\t%s\n' ${found_pods})
        "
  fi
}

trap do_teardown EXIT

# Get node IP addresses and pick one as our test point.
detect-minions
test_node="${MINION_NAMES[0]}"
master="${MASTER_NAME}"

# Launch some pods and services.
svc1_name="service-${RANDOM}"
svc1_port=80
svc1_count=3
svc1_publics="192.168.1.1 192.168.1.2"
start_service "${svc1_name}" "${svc1_port}" "${svc1_count}" "${svc1_publics}"

svc2_name="service-${RANDOM}"
svc2_port=80
svc2_count=3
start_service "${svc2_name}" "${svc2_port}" "${svc2_count}"

# Wait for the pods to become "running".
wait_for_pods "${svc1_name}" "${svc1_count}"
wait_for_pods "${svc2_name}" "${svc2_count}"

# Get the sorted lists of pods.
svc1_pods=$(query_pods "${svc1_name}" "${svc1_count}")
svc2_pods=$(query_pods "${svc2_name}" "${svc2_count}")

# Get the portal IPs.
svc1_ip=$(${KUBECTL} get services -o template '--template={{.spec.portalIP}}' "${svc1_name}" --api-version=v1beta3)
test -n "${svc1_ip}" || error "Service1 IP is blank"
svc2_ip=$(${KUBECTL} get services -o template '--template={{.spec.portalIP}}' "${svc2_name}" --api-version=v1beta3)
test -n "${svc2_ip}" || error "Service2 IP is blank"
if [[ "${svc1_ip}" == "${svc2_ip}" ]]; then
  error "Portal IPs conflict: ${svc1_ip}"
fi

#
# Test 1: Prove that the service portal is alive.
#
echo "Test 1: Prove that the service portal is alive."
echo "Verifying the portals from the host"
wait_for_service_up "${svc1_name}" "${svc1_ip}" "${svc1_port}" \
    "${svc1_count}" "${svc1_pods}"
for ip in ${svc1_publics}; do
  wait_for_service_up "${svc1_name}" "${ip}" "${svc1_port}" \
      "${svc1_count}" "${svc1_pods}"
done
wait_for_service_up "${svc2_name}" "${svc2_ip}" "${svc2_port}" \
    "${svc2_count}" "${svc2_pods}"
echo "Verifying the portals from a container"
verify_from_container "${svc1_name}" "${svc1_ip}" "${svc1_port}" \
    "${svc1_count}" "${svc1_pods}"
for ip in ${svc1_publics}; do
  verify_from_container "${svc1_name}" "${ip}" "${svc1_port}" \
      "${svc1_count}" "${svc1_pods}"
done
verify_from_container "${svc2_name}" "${svc2_ip}" "${svc2_port}" \
    "${svc2_count}" "${svc2_pods}"

#
# Test 2: Bounce the proxy and make sure the portal comes back.
#
echo "Test 2: Bounce the proxy and make sure the portal comes back."
echo "Restarting kube-proxy"
restart-kube-proxy "${test_node}"
echo "Verifying the portals from the host"
wait_for_service_up "${svc1_name}" "${svc1_ip}" "${svc1_port}" \
    "${svc1_count}" "${svc1_pods}"
wait_for_service_up "${svc2_name}" "${svc2_ip}" "${svc2_port}" \
    "${svc2_count}" "${svc2_pods}"
echo "Verifying the portals from a container"
verify_from_container "${svc1_name}" "${svc1_ip}" "${svc1_port}" \
    "${svc1_count}" "${svc1_pods}"
verify_from_container "${svc2_name}" "${svc2_ip}" "${svc2_port}" \
    "${svc2_count}" "${svc2_pods}"

#
# Test 3: Stop one service and make sure it is gone.
#
echo "Test 3: Stop one service and make sure it is gone."
stop_service "${svc1_name}"
wait_for_service_down "${svc1_name}" "${svc1_ip}" "${svc1_port}"

#
# Test 4: Bring up another service.
# TODO: Actually add a test to force re-use.
#
echo "Test 4: Bring up another service."
svc3_name="service3"
svc3_port=80
svc3_count=3
start_service "${svc3_name}" "${svc3_port}" "${svc3_count}"

# Wait for the pods to become "running".
wait_for_pods "${svc3_name}" "${svc3_count}"

# Get the sorted lists of pods.
svc3_pods=$(query_pods "${svc3_name}" "${svc3_count}")

# Get the portal IP.
svc3_ip=$(${KUBECTL} get services -o template '--template={{.spec.portalIP}}' "${svc3_name}" --api-version=v1beta3)
test -n "${svc3_ip}" || error "Service3 IP is blank"

echo "Verifying the portals from the host"
wait_for_service_up "${svc3_name}" "${svc3_ip}" "${svc3_port}" \
    "${svc3_count}" "${svc3_pods}"
echo "Verifying the portals from a container"
verify_from_container "${svc3_name}" "${svc3_ip}" "${svc3_port}" \
    "${svc3_count}" "${svc3_pods}"

#
# Test 5: Remove the iptables rules, make sure they come back.
#
echo "Test 5: Remove the iptables rules, make sure they come back."
echo "Manually removing iptables rules"
# Remove both the new and old style chains, in case we're testing on an old kubelet
ssh-to-node "${test_node}" "sudo iptables -t nat -F KUBE-PORTALS-HOST || true"
ssh-to-node "${test_node}" "sudo iptables -t nat -F KUBE-PORTALS-CONTAINER || true"
ssh-to-node "${test_node}" "sudo iptables -t nat -F KUBE-PROXY || true"
echo "Verifying the portals from the host"
wait_for_service_up "${svc3_name}" "${svc3_ip}" "${svc3_port}" \
    "${svc3_count}" "${svc3_pods}"
echo "Verifying the portals from a container"
verify_from_container "${svc3_name}" "${svc3_ip}" "${svc3_port}" \
    "${svc3_count}" "${svc3_pods}"

#
# Test 6: Restart the master, make sure portals come back.
#
echo "Test 6: Restart the master, make sure portals come back."
echo "Restarting the master"
restart-apiserver "${master}"
sleep 5
echo "Verifying the portals from the host"
wait_for_service_up "${svc3_name}" "${svc3_ip}" "${svc3_port}" \
    "${svc3_count}" "${svc3_pods}"
echo "Verifying the portals from a container"
verify_from_container "${svc3_name}" "${svc3_ip}" "${svc3_port}" \
    "${svc3_count}" "${svc3_pods}"

#
# Test 7: Bring up another service, make sure it does not re-use Portal IPs.
#
echo "Test 7: Bring up another service, make sure it does not re-use Portal IPs."
svc4_name="service4"
svc4_port=80
svc4_count=3
start_service "${svc4_name}" "${svc4_port}" "${svc4_count}"

# Wait for the pods to become "running".
wait_for_pods "${svc4_name}" "${svc4_count}"

# Get the sorted lists of pods.
svc4_pods=$(query_pods "${svc4_name}" "${svc4_count}")

# Get the portal IP.
svc4_ip=$(${KUBECTL} get services -o template '--template={{.spec.portalIP}}' "${svc4_name}" --api-version=v1beta3)
test -n "${svc4_ip}" || error "Service4 IP is blank"
if [[ "${svc4_ip}" == "${svc2_ip}" || "${svc4_ip}" == "${svc3_ip}" ]]; then
  error "Portal IPs conflict: ${svc4_ip}"
fi

echo "Verifying the portals from the host"
wait_for_service_up "${svc4_name}" "${svc4_ip}" "${svc4_port}" \
    "${svc4_count}" "${svc4_pods}"
echo "Verifying the portals from a container"
verify_from_container "${svc4_name}" "${svc4_ip}" "${svc4_port}" \
    "${svc4_count}" "${svc4_pods}"

# TODO: test createExternalLoadBalancer

exit 0
