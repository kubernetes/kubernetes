#!/usr/bin/env bash

# Copyright 2018 The Kubernetes Authors.
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

set -o errexit
set -o nounset
set -o pipefail

create_test_pods(){
  # create test pods we can work with
  kubectl create -f - "${kube_flags[@]:?}" << __EOF__
{
  "kind": "Pod",
  "apiVersion": "v1",
  "metadata": {
    "name": "test-pod-1",
    "labels": {
      "e": "f",
      "type": "test-pod"
    }
  },
  "spec": {
    "nodeName": "127.0.0.1",
    "containers": [
      {
        "name": "container-1",
        "resources": {},
        "image": "test-image"
      }
    ]
  }
}
__EOF__

  kubectl create -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "Pod",
  "apiVersion": "v1",
  "metadata": {
    "name": "test-pod-2",
    "labels": {
      "c": "d",
      "type": "test-pod"
    }
  },
  "spec": {
    "nodeName": "127.0.0.1",
    "containers": [
      {
        "name": "container-1",
        "resources": {},
        "image": "test-image"
      }
    ]
  }
}
__EOF__
}

delete_test_pods() {
  # need to use --force because node is unready
  kubectl delete pod/test-pod-1 --force --ignore-not-found
  kubectl delete pod/test-pod-2 --force --ignore-not-found
}

run_cluster_management_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing cluster-management commands"

  kube::test::get_object_assert nodes "{{range.items}}{{${id_field:?}}}:{{end}}" '127.0.0.1:'

  create_test_pods

  # taint/untaint
  # Pre-condition: node doesn't have dedicated=foo:PreferNoSchedule taint
  kube::test::get_object_assert "nodes 127.0.0.1" "{{range .spec.taints}}{{if eq .key \"dedicated\"}}{{.key}}={{.value}}:{{.effect}}{{end}}{{end}}" "" # expect no output
  # Dry-run
  kubectl taint node 127.0.0.1 --dry-run=client dedicated=foo:PreferNoSchedule
  kubectl taint node 127.0.0.1 --dry-run=server dedicated=foo:PreferNoSchedule
  kube::test::get_object_assert "nodes 127.0.0.1" "{{range .spec.taints}}{{if eq .key \"dedicated\"}}{{.key}}={{.value}}:{{.effect}}{{end}}{{end}}" "" # expect no output
  # taint can add a taint (<key>=<value>:<effect>)
  kubectl taint node 127.0.0.1 dedicated=foo:PreferNoSchedule
  kube::test::get_object_assert "nodes 127.0.0.1" "{{range .spec.taints}}{{if eq .key \"dedicated\"}}{{.key}}={{.value}}:{{.effect}}{{end}}{{end}}" "dedicated=foo:PreferNoSchedule"
  # taint can remove a taint
  kubectl taint node 127.0.0.1 dedicated-
  # taint can add a taint (<key>:<effect>)
  kubectl taint node 127.0.0.1 dedicated:PreferNoSchedule
  kube::test::get_object_assert "nodes 127.0.0.1" "{{range .spec.taints}}{{if eq .key \"dedicated\"}}{{.key}}={{.value}}:{{.effect}}{{end}}{{end}}" "dedicated=<no value>:PreferNoSchedule"
  # Node has field manager for kubectl taint
  output_message=$(kubectl get node 127.0.0.1 --show-managed-fields -o=jsonpath='{.metadata.managedFields[*].manager}' "${kube_flags[@]:?}" 2>&1)
  kube::test::if_has_string "${output_message}" 'kubectl-taint'
  # Dry-run remove a taint
  kubectl taint node 127.0.0.1 --dry-run=client dedicated-
  kubectl taint node 127.0.0.1 --dry-run=server dedicated-
  kube::test::get_object_assert "nodes 127.0.0.1" "{{range .spec.taints}}{{if eq .key \"dedicated\"}}{{.key}}={{.value}}:{{.effect}}{{end}}{{end}}" "dedicated=<no value>:PreferNoSchedule"
  # taint can remove a taint
  kubectl taint node 127.0.0.1 dedicated-
  # Post-condition: node doesn't have dedicated=foo:PreferNoSchedule taint
  kube::test::get_object_assert "nodes 127.0.0.1" "{{range .spec.taints}}{{if eq .key \"dedicated\"}}{{.key}}={{.value}}:{{.effect}}{{end}}{{end}}" "" # expect no output

  ### kubectl cordon update with --dry-run does not mark node unschedulable
  # Pre-condition: node is schedulable
  kube::test::get_object_assert "nodes 127.0.0.1" "{{.spec.unschedulable}}" '<no value>'
  kubectl cordon "127.0.0.1" --dry-run=client
  kubectl cordon "127.0.0.1" --dry-run=server
  kube::test::get_object_assert "nodes 127.0.0.1" "{{.spec.unschedulable}}" '<no value>'

  ### kubectl drain update with --dry-run does not mark node unschedulable
  # Pre-condition: node is schedulable
  kube::test::get_object_assert "nodes 127.0.0.1" "{{.spec.unschedulable}}" '<no value>'
  kubectl drain "127.0.0.1" --dry-run=client --force
  kubectl drain "127.0.0.1" --dry-run=server --force
  # Post-condition: node still exists, node is still schedulable
  kube::test::get_object_assert nodes "{{range.items}}{{$id_field}}:{{end}}" '127.0.0.1:'
  kube::test::get_object_assert "nodes 127.0.0.1" "{{.spec.unschedulable}}" '<no value>'

  ### kubectl drain with --pod-selector only evicts pods that match the given selector
  # Pre-condition: node is schedulable
  kube::test::get_object_assert "nodes 127.0.0.1" "{{.spec.unschedulable}}" '<no value>'
  # Pre-condition: test-pod-1 and test-pod-2 exist
  kube::test::get_object_assert "pods" "{{range .items}}{{.metadata.name}},{{end}}" 'test-pod-1,test-pod-2,'
  # dry-run command
  kubectl drain "127.0.0.1" --pod-selector 'e in (f)' --dry-run=client --force
  kubectl drain "127.0.0.1" --pod-selector 'e in (f)' --dry-run=server --force
  kube::test::get_object_assert "pods" "{{range .items}}{{.metadata.name}},{{end}}" 'test-pod-1,test-pod-2,'
  # command - need --force because pod is unmanaged and --skip-wait-for-delete-timeout because node is unready
  response=$(kubectl drain "127.0.0.1" --force --pod-selector 'e in (f)' --skip-wait-for-delete-timeout=1)
  kube::test::if_has_string "${response}" "evicting pod .*/test-pod-1"
  # only "test-pod-1" should have been matched and deleted - test-pod-2 should not have a deletion timestamp
  kube::test::get_object_assert "pods/test-pod-2" "{{.metadata.deletionTimestamp}}" '<no value>'
  # Post-condition: recreate test pods -- they have deletionTimestamp set but will not go away because node is unready
  delete_test_pods
  create_test_pods
  # Post-condition: node is schedulable
  kubectl uncordon "127.0.0.1"
  kube::test::get_object_assert "nodes 127.0.0.1" "{{.spec.unschedulable}}" '<no value>'

  ### kubectl uncordon update with --dry-run is a no-op
  # Pre-condition: node is already schedulable
  kube::test::get_object_assert "nodes 127.0.0.1" "{{.spec.unschedulable}}" '<no value>'
  response=$(kubectl uncordon "127.0.0.1" --dry-run=client)
  kube::test::if_has_string "${response}" 'already uncordoned'
  response=$(kubectl uncordon "127.0.0.1" --dry-run=server)
  kube::test::if_has_string "${response}" 'already uncordoned'
  # Post-condition: node is still schedulable
  kube::test::get_object_assert "nodes 127.0.0.1" "{{.spec.unschedulable}}" '<no value>'

  ### kubectl drain command fails when both --selector and a node argument are given
  # Pre-condition: node exists and contains label test=label
  kubectl label node "127.0.0.1" "test=label"
  kube::test::get_object_assert "nodes 127.0.0.1" '{{.metadata.labels.test}}' 'label'
  response=$(! kubectl drain "127.0.0.1" --selector test=label 2>&1)
  kube::test::if_has_string "${response}" 'cannot specify both a node name'

   ### Test kubectl drain chunk size
   # Pre-condition: node exists and contains label test=label
   kube::test::get_object_assert "nodes 127.0.0.1" '{{.metadata.labels.test}}' 'label'
   # Pre-condition: node is schedulable
   kube::test::get_object_assert "nodes 127.0.0.1" "{{.spec.unschedulable}}" '<no value>'
   # Pre-condition: test-pod-1 and test-pod-2 exist
   kube::test::get_object_assert "pods" "{{range .items}}{{.metadata.name}},{{end}}" 'test-pod-1,test-pod-2,'
   # command - need to use force because pods are unmanaged, dry run (or skip-wait) because node is unready
   output_message=$(kubectl --v=6 drain --force --pod-selector type=test-pod --selector test=label --chunk-size=1 --dry-run=client 2>&1 "${kube_flags[@]}")
   # Post-condition: Check if we get a limit on node, and both limit and continue on pods
   kube::test::if_has_string "${output_message}" '"Response" verb="GET" url=".*/v1/nodes?labelSelector=test%3Dlabel&limit=1" status="200 OK"'
   kube::test::if_has_string "${output_message}" '"Response" verb="GET" url=".*/v1/pods?fieldSelector=spec.nodeName%3D127.0.0.1&labelSelector=type%3Dtest-pod&limit=1" status="200 OK"'
   kube::test::if_has_string "${output_message}" '"Response" verb="GET" url=".*/v1/pods?continue=.*&fieldSelector=spec.nodeName%3D127.0.0.1&labelSelector=type%3Dtest-pod&limit=1" status="200 OK"'
   # Post-condition: Check we evict multiple pages worth of pods
   kube::test::if_has_string "${output_message}" "evicting pod .*/test-pod-1"
   kube::test::if_has_string "${output_message}" "evicting pod .*/test-pod-2"
   # Post-condition: node is schedulable
   kubectl uncordon "127.0.0.1"
   kube::test::get_object_assert "nodes 127.0.0.1" "{{.spec.unschedulable}}" '<no value>'

   ### Test kubectl drain chunk size defaults to 500
   output_message=$(kubectl --v=6 drain --force --selector test=label --dry-run=client 2>&1 "${kube_flags[@]}")
   # Post-condition: Check if we get a limit
   kube::test::if_has_string "${output_message}" '"Response" verb="GET" url=".*/v1/nodes?labelSelector=test%3Dlabel&limit=500" status="200 OK"'
   kube::test::if_has_string "${output_message}" '"Response" verb="GET" url=".*/v1/pods?fieldSelector=spec.nodeName%3D127.0.0.1&limit=500" status="200 OK"'

  ### kubectl cordon command fails when no arguments are passed
  # Pre-condition: node exists
  response=$(! kubectl cordon 2>&1)
  kube::test::if_has_string "${response}" 'error\: USAGE\: cordon NODE'

  ### kubectl cordon selects no nodes with an empty --selector=
  # Pre-condition: node "127.0.0.1" is uncordoned
  kubectl uncordon "127.0.0.1"
  response=$(! kubectl cordon --selector= 2>&1)
  kube::test::if_has_string "${response}" 'must provide one or more resources'
  # test=label matches our node
  response=$(kubectl cordon --selector test=label)
  kube::test::if_has_string "${response}" 'node/127.0.0.1 cordoned'
  # invalid=label does not match any nodes
  response=$(kubectl cordon --selector invalid=label)
  kube::test::if_has_not_string "${response}" 'cordoned'
  # Post-condition: node "127.0.0.1" is cordoned
  kube::test::get_object_assert "nodes 127.0.0.1" "{{.spec.unschedulable}}" 'true'

  # Clean up test pods
  delete_test_pods

  set +o nounset
  set +o errexit
}
