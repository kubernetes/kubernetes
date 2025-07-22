#!/usr/bin/env bash

# Copyright 2019 The Kubernetes Authors.
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

run_kubectl_exec_pod_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl exec POD COMMAND"

  ### Test execute non-existing POD
  output_message=$(! kubectl exec abc 2>&1 -- date)
  # POD abc should error since it doesn't exist
  kube::test::if_has_string "${output_message}" 'pods "abc" not found'

  ### Test execute multiple resources
  output_message=$(! kubectl exec -f - 2>&1 -- echo test << __EOF__
apiVersion: v1
kind: Pod
metadata:
  name: test
spec:
  containers:
  - name: nginx
    image: nginx
---
apiVersion: v1
kind: Pod
metadata:
  name: test2
spec:
  containers:
  - name: nginx
    image: nginx
__EOF__
)
  kube::test::if_has_string "${output_message}" 'cannot exec into multiple objects at a time'

  ### Test execute existing POD
  # Create test-pod
  kubectl create -f hack/testdata/pod.yaml
  # Execute existing POD
  output_message=$(! kubectl exec test-pod date 2>&1)
  # POD test-pod is exists this is shouldn't have output not found
  kube::test::if_has_not_string "${output_message}" 'pods "test-pod" not found'
  # These must be pass the validate
  kube::test::if_has_not_string "${output_message}" 'pod or type/name must be specified'

  # Clean up
  kubectl delete pods test-pod

  set +o nounset
  set +o errexit
}

run_kubectl_exec_resource_name_tests() {
  set +o nounset
  set +o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl exec TYPE/NAME COMMAND"

  ### Test execute invalid resource type
  output_message=$(! kubectl exec foo/bar date 2>&1)
  # resource type foo should error since it's invalid
  kube::test::if_has_string "${output_message}" 'error:'

  ### Test execute non-existing resources
  output_message=$(! kubectl exec deployments/bar date 2>&1)
  # resource type foo should error since it doesn't exist
  kube::test::if_has_string "${output_message}" '"bar" not found'

  kubectl create -f hack/testdata/pod.yaml
  kubectl create -f hack/testdata/frontend-replicaset.yaml
  kubectl create -f hack/testdata/configmap.yaml

  ### Test execute non-implemented resources
  output_message=$(! kubectl exec configmap/test-set-env-config date 2>&1)
  # resource type configmap should error since configmap not implemented to be attached
  kube::test::if_has_string "${output_message}" 'not implemented'

  ### Test execute exists and valid resource type.
  # Just check the output, since test-cmd not run kubelet, pod never be assigned.
  # and not really can run `kubectl exec` command

  output_message=$(! kubectl exec pods/test-pod date 2>&1)
  # POD test-pod is exists this is shouldn't have output not found
  kube::test::if_has_not_string "${output_message}" 'not found'
  # These must be pass the validate
  kube::test::if_has_not_string "${output_message}" 'pod, type/name or --filename must be specified'

  output_message=$(! kubectl exec replicaset/frontend date 2>&1)
  # Replicaset frontend is valid and exists will select the first pod.
  # and Shouldn't have output not found
  kube::test::if_has_not_string "${output_message}" 'not found'
  # These must be pass the validate
  kube::test::if_has_not_string "${output_message}" 'pod, type/name or --filename must be specified'

  # Clean up
  kubectl delete pods/test-pod
  kubectl delete replicaset/frontend
  kubectl delete configmap/test-set-env-config

  set +o nounset
  set +o errexit
}
