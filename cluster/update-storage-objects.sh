#!/usr/bin/env bash

# Copyright 2015 The Kubernetes Authors.
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

# Script to update etcd objects as per the latest API Version.
# This just reads all objects and then writes them back as is to ensure that
# they are written using the latest API version.
#
# Steps to use this script to upgrade the cluster to a new version:
# https://kubernetes.io/docs/tasks/administer-cluster/cluster-management/#upgrading-to-a-different-api-version

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

KUBECTL="${KUBE_OUTPUT_HOSTBIN}/kubectl"

# List of resources to be updated.
# TODO: Get this list of resources from server once
# http://issue.k8s.io/2057 is fixed.
declare -a resources=(
    "endpoints"
    "events"
    "limitranges"
    "namespaces"
    "nodes"
    "pods"
    "persistentvolumes"
    "persistentvolumeclaims"
    "replicationcontrollers"
    "resourcequotas"
    "secrets"
    "services"
    "jobs"
    "horizontalpodautoscalers"
    "storageclasses"
    "roles.rbac.authorization.k8s.io"
    "rolebindings.rbac.authorization.k8s.io"
    "clusterroles.rbac.authorization.k8s.io"
    "clusterrolebindings.rbac.authorization.k8s.io"
    "networkpolicies.networking.k8s.io"
)

# Find all the namespaces.
IFS=" " read -r -a namespaces <<< "$("${KUBECTL}" get namespaces -o go-template="{{range.items}}{{.metadata.name}} {{end}}")"
if [ -z "${namespaces:-}" ]
then
  echo "Unexpected: No namespace found. Nothing to do."
  exit 1
fi

all_failed=1

for resource in "${resources[@]}"
do
  for namespace in "${namespaces[@]}"
  do
    # If get fails, assume it's because the resource hasn't been installed in the apiserver.
    # TODO hopefully we can remove this once we use dynamic discovery of gettable/updateable
    # resources.
    set +e
    IFS=" " read -r -a instances <<< "$("${KUBECTL}" get "${resource}" --namespace="${namespace}" -o go-template="{{range.items}}{{.metadata.name}} {{end}}")"
    result=$?
    set -e

    if [[ "${all_failed}" -eq 1 && "${result}" -eq 0 ]]; then
      all_failed=0
    fi

    # Nothing to do if there is no instance of that resource.
    if [[ -z "${instances:-}" ]]
    then
      continue
    fi
    for instance in "${instances[@]}"
    do
      # Read and then write it back as is.
      # Update can fail if the object was updated after we fetched the
      # object, but before we could update it. We, hence, try the update
      # operation multiple times. But 5 continuous failures indicate some other
      # problem.
      success=0
      for (( tries=0; tries<5; ++tries ))
      do
        filename="/tmp/k8s-${namespace}-${resource}-${instance}.json"
        ( "${KUBECTL}" get "${resource}" "${instance}" --namespace="${namespace}" -o json > "${filename}" ) || true
        if [[ ! -s "${filename}" ]]
        then
          # This happens when the instance has been deleted. We can hence ignore
          # this instance.
          echo "Looks like ${instance} got deleted. Ignoring it"
          success=1
          break
        fi
        output=$("${KUBECTL}" replace -f "${filename}" --namespace="${namespace}") || true
        rm "${filename}"
        if [ -n "${output:-}" ]
        then
          success=1
          break
        fi
      done
      if [[ "${success}" -eq 0 ]]
      then
        echo "Error: failed to update ${resource}/${instance} in ${namespace} namespace after 5 tries"
        exit 1
      fi
    done
    if [[ "${resource}" == "namespaces" ]] || [[ "${resource}" == "nodes" ]]
    then
      # These resources are namespace agnostic. No need to update them for every
      # namespace.
      break
    fi
  done
done

if [[ "${all_failed}" -eq 1 ]]; then
  echo "kubectl get failed for all resources"
  exit 1
fi

echo "All objects updated successfully!!"

exit 0
