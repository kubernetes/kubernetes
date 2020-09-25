#!/usr/bin/env bash

# Copyright 2020 The Kubernetes Authors.
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

# These tests enforce behavior of kube-addon-manager functions against a real
# cluster. A working Kubernetes cluster must be set up with kubectl configured.
# To run with the released version of kubectl, use `make test`.

set -o errexit
set -o pipefail
set -o nounset

# Default kubectl to the test users installation if needed.
KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"

# Disabling shellcheck following files as the full path would be required.
# shellcheck disable=SC1091
source "kube-addons.sh"

TEST_NS="kube-addon-manager-test"

function retry() {
  local tries=10
  while [ "${tries}" -gt 0 ]; do
    "$@" && return 0;
    (( tries-- ))
    sleep 1
  done
}

function setup(){
  retry kubectl create namespace "${TEST_NS}"
}

function teardown() {
  retry kubectl delete namespace "${TEST_NS}"
}

function error() {
  echo -e "\e[31m$*\e[0m"
}

function echo_green() {
  echo -e "\e[32m$*\e[0m"
}

function echo_blue() {
  echo -e "\e[34m$*\e[0m"
}

function test_create_resource_reconcile() {
  local limitrange
  read -r -d '' limitrange << EOF
apiVersion: "v1"
kind: "LimitRange"
metadata:
  name: "limits"
  namespace: "${TEST_NS}"
  labels:
    addonmanager.kubernetes.io/mode: Reconcile
spec:
  limits:
    - type: "Container"
      defaultRequest:
        cpu: "100m"
EOF

  # arguments are yaml text, number of tries, delay, name of file, and namespace
  echo_blue "Creating initial resource to test Reconcile mode"
  create_resource_from_string "${limitrange}" "10" "1" "limitrange.yaml" "${TEST_NS}"
  if ! (kubectl get limits/limits -n "${TEST_NS}"); then
    error "failed to create limits w/ reconcile"
    return 1
  elif ! (kubectl get limits/limits -n ${TEST_NS} -oyaml | grep --silent "100m"); then
    error "limits does not match applied config"
    return 1
  fi

  # Changes to addons with mode reconcile should be reflected.
  echo_blue "Changes to manifest should be reflected in the cluster"
  limitrange="${limitrange//100m/50m}"
  create_resource_from_string "${limitrange}" "10" "1" "limitrange.yaml" "${TEST_NS}"
  if kubectl get limits/limits -n ${TEST_NS} -oyaml | grep --silent "100m"; then
    error "failed to update resource, still has 100m"
    return 1
  elif ! (kubectl get limits/limits -n ${TEST_NS} -oyaml | grep --silent "50m"); then
    error "failed to update resource, 50m limit was not reflected"
    return 1
  fi

  # Finally, the users configuration will not be respected.
  echo_blue "Changes the user makes should be overwritten by kube-addon-manager"
  EDITOR="sed -i 's/50m/600m/'" kubectl edit limits/limits -n ${TEST_NS}
  if kubectl get limits/limits -n ${TEST_NS} -oyaml | grep --silent "50m"; then
    error "failed to edit resource with sed -- test is broken"
    return 1
  fi
  create_resource_from_string "${limitrange}" "10" "1" "limitrange.yaml" "${TEST_NS}"
  if ! ( kubectl get limits/limits -n ${TEST_NS} -oyaml | grep --silent "50m"); then
    error "failed to update resource, user config was respected when it should have been rewritten"
    return 1
  fi
}

function test_create_resource_ensureexists() {
  local limitrange
  read -r -d '' limitrange << EOF
apiVersion: "v1"
kind: "LimitRange"
metadata:
  name: "limits"
  namespace: "${TEST_NS}"
  labels:
    addonmanager.kubernetes.io/mode: EnsureExists
spec:
  limits:
    - type: "Container"
      defaultRequest:
        cpu: "100m"
EOF

  # arguments are yaml text, number of tries, delay, name of file, and namespace
  echo_blue "Creating initial resource to test mode EnsureExists"
  create_resource_from_string "${limitrange}" "10" "1" "limitrange.yaml" "${TEST_NS}"
  if ! (kubectl get limits/limits -n "${TEST_NS}"); then
    error "failed to create limits w/ EnsureExists"
    return 1
  elif ! (kubectl get limits/limits -n ${TEST_NS} -oyaml | grep --silent "100m"); then
    error "limits does not match applied config"
    return 1
  fi

  # Changes to addons with mode EnsureExists should NOT be reflected.
  echo_blue "Changes to the manifest should not be reconciled with the cluster"
  limitrange="${limitrange//100m/50m}"
  create_resource_from_string "${limitrange}" "10" "1" "limitrange.yaml" "${TEST_NS}"
  if kubectl get limits/limits -n ${TEST_NS} -oyaml | grep --silent "50m"; then
    error "failed to respect existing resource, was overwritten despite EnsureExists"
    return 1
  fi

  # the users configuration must be respected
  echo_blue "User configuration will be persisted for EnsureExists"
  EDITOR="sed -i 's/100m/600m/'" kubectl edit limits/limits -n ${TEST_NS}
  if kubectl get limits/limits -n ${TEST_NS} -oyaml | grep --silent "100m"; then
    error "failed to edit resource with sed -- test is broken"
    return 1
  fi
  create_resource_from_string "${limitrange}" "10" "1" "limitrange.yaml" "${TEST_NS}"
  if kubectl get limits/limits -n ${TEST_NS} -oyaml | grep --silent "100m"; then
    error "failed to respect user changes to EnsureExists object"
    return 1
  fi

  # unless they delete the object, in which case it should return
  echo_blue "Missing EnsureExists resources will be re-created"
  kubectl delete limits/limits -n ${TEST_NS}
  if kubectl get limits/limits -n ${TEST_NS}; then
    error "failed to delete limitrange"
    return 1
  fi
  create_resource_from_string "${limitrange}" "10" "1" "limitrange.yaml" "${TEST_NS}"
  if ! kubectl get limits/limits -n ${TEST_NS}; then
    error "failed to recreate deleted EnsureExists resource"
    return 1
  fi
}

function test_create_multiresource() {
  local limitrange
  read -r -d '' limitrange << EOF
apiVersion: "v1"
kind: "LimitRange"
metadata:
  name: "limits"
  namespace: "${TEST_NS}"
  labels:
    addonmanager.kubernetes.io/mode: EnsureExists
spec:
  limits:
    - type: "Container"
      defaultRequest:
        cpu: "100m"
---
apiVersion: "v1"
kind: "LimitRange"
metadata:
  name: "limits2"
  namespace: "${TEST_NS}"
  labels:
    addonmanager.kubernetes.io/mode: Reconcile
spec:
  limits:
    - type: "Container"
      defaultRequest:
        cpu: "100m"
EOF

  # arguments are yaml text, number of tries, delay, name of file, and namespace
  echo_blue "Creating initial resources from multi-resource manifest"
  create_resource_from_string "${limitrange}" "10" "1" "limitrange.yaml" "${TEST_NS}"
  if ! (kubectl get limits/limits -n "${TEST_NS}"); then
    error "failed to create limits w/ EnsureExists"
    return 1
  elif ! (kubectl get limits/limits2 -n "${TEST_NS}"); then
    error "failed to create limits2 w/ Reconcile"
    return 1
  fi

  # Changes to addons with mode EnsureExists should NOT be reflected.
  # However, the mode=Reconcile addon should be changed.
  echo_blue "Multi-resource manifest changes should apply to EnsureExists, not Reconcile"
  limitrange="${limitrange//100m/50m}"
  create_resource_from_string "${limitrange}" "10" "1" "limitrange.yaml" "${TEST_NS}"
  if kubectl get limits/limits -n ${TEST_NS} -oyaml | grep --silent "50m"; then
    error "failed to respect existing resource, was overwritten despite EnsureExists"
    return 1
  elif kubectl get limits/limits2 -n ${TEST_NS} | grep --silent "100m"; then
    error "failed to update resource with mode Reconcile"
    return 1
  fi

  # the users configuration must be respected for EnsureExists
  echo_blue "Multi-resource manifest should not overwrite user config in EnsureExists"
  EDITOR="sed -i 's/100m/600m/'" kubectl edit limits/limits -n ${TEST_NS}
  if kubectl get limits/limits -n ${TEST_NS} -oyaml | grep --silent "100m"; then
    error "failed to edit resource with sed -- test is broken"
    return 1
  fi
  create_resource_from_string "${limitrange}" "10" "1" "limitrange.yaml" "${TEST_NS}"
  if kubectl get limits/limits -n ${TEST_NS} -oyaml | grep --silent "100m"; then
    error "failed to respect user changes to EnsureExists object"
    return 1
  fi

  # But not for Reconcile.
  echo_blue "Multi-resource manifest should overwrite user config in EnsureExists"
  EDITOR="sed -i 's/50m/600m/'" kubectl edit limits/limits2 -n ${TEST_NS}
  if kubectl get limits/limits2 -n ${TEST_NS} -oyaml | grep --silent "50m"; then
    error "failed to edit resource with sed -- test is broken"
    return 1
  fi
  create_resource_from_string "${limitrange}" "10" "1" "limitrange.yaml" "${TEST_NS}"
  if ! ( kubectl get limits/limits2 -n ${TEST_NS} -oyaml | grep --silent "50m"); then
    error "failed to update resource, user config was respected when it should have been rewritten"
    return 1
  fi
}

function test_func() {
  local -r name="${1}"

  echo_blue "=== TEST ${name}"
  setup
  if ! "${name}"; then
    failures=$((failures+1))
    error "=== FAIL"
  else
    echo_green "=== PASS"
  fi
  teardown
}

failures=0
test_func test_create_resource_reconcile
test_func test_create_resource_ensureexists
test_func test_create_multiresource
if [ "${failures}" -gt 0 ]; then
  error "no. failed tests: ${failures}"
  error "FAIL"
  exit 1
else
  echo_green "PASS"
fi
