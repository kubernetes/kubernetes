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

run_initializer_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing --include-uninitialized"

  ### Create a deployment
  kubectl create --request-timeout=1 -f hack/testdata/initializer-deployments.yaml 2>&1 "${kube_flags[@]}" || true

  ### Test kubectl get --include-uninitialized
  # Command
  output_message=$(kubectl get deployments 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "No resources found" should be part of the output
  kube::test::if_has_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl get deployments --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "No resources found" should be part of the output
  kube::test::if_has_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl get deployments --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: I assume "web" is the deployment name
  kube::test::if_has_string "${output_message}" 'web'
  # Command
  output_message=$(kubectl get deployments web 2>&1 "${kube_flags[@]}")
  # Post-condition: I assume "web" is the deployment name
  kube::test::if_has_string "${output_message}" 'web'
  # Post-condition: The text "No resources found" should be part of the output
  kube::test::if_has_string "${output_message}" 'No resources found'

  ### Test kubectl describe --include-uninitialized
  # Command
  output_message=$(kubectl describe deployments 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "run=web" should be part of the output
  kube::test::if_has_string "${output_message}" 'run=web'
  # Command
  output_message=$(kubectl describe deployments --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "run=web" should be part of the output
  kube::test::if_has_string "${output_message}" 'run=web'
  # Command
  output_message=$(kubectl describe deployments --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl describe deployments web --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "run=web" should be part of the output
  kube::test::if_has_string "${output_message}" 'run=web'
  # Command
  output_message=$(kubectl describe deployments web --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "run=web" should be part of the output
  kube::test::if_has_string "${output_message}" 'run=web'

  ### Test kubectl label --include-uninitialized
  # Command
  output_message=$(kubectl label deployments labelkey1=labelvalue1 --all 2>&1 "${kube_flags[@]}")
  # Post-condition: web is labelled
  kube::test::if_has_string "${output_message}" 'deployment "web" labeled'
  kube::test::get_object_assert 'deployments web' "{{${labels_field}.labelkey1}}" 'labelvalue1'
  # Command
  output_message=$(kubectl label deployments labelkey2=labelvalue2 --all --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl label deployments labelkey3=labelvalue3 -l run=web 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl label deployments labelkey4=labelvalue4 -l run=web --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: web is labelled
  kube::test::if_has_string "${output_message}" 'deployment "web" labeled'
  kube::test::get_object_assert 'deployments web' "{{${labels_field}.labelkey4}}" 'labelvalue4'
  # Command
  output_message=$(kubectl label deployments labelkey5=labelvalue5 -l run=web --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl label deployments labelkey6=labelvalue6 -l run=web --all --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: web is labelled
  kube::test::if_has_string "${output_message}" 'deployment "web" labeled'
  kube::test::get_object_assert 'deployments web' "{{${labels_field}.labelkey6}}" 'labelvalue6'
  # Command
  output_message=$(kubectl label deployments web labelkey7=labelvalue7 2>&1 "${kube_flags[@]}")
  # Post-condition: web is labelled
  kube::test::if_has_string "${output_message}" 'deployment "web" labeled'
  kube::test::get_object_assert 'deployments web' "{{${labels_field}.labelkey7}}" 'labelvalue7'
  # Found All Labels
  kube::test::get_object_assert 'deployments web' "{{${labels_field}}}" 'map[labelkey1:labelvalue1 labelkey4:labelvalue4 labelkey6:labelvalue6 labelkey7:labelvalue7 run:web]'

  ### Test kubectl annotate --include-uninitialized
  # Command
  output_message=$(kubectl annotate deployments annotatekey1=annotatevalue1 --all 2>&1 "${kube_flags[@]}")
  # Post-condition: DEPLOYMENT has annotation
  kube::test::if_has_string "${output_message}" 'deployment "web" annotated'
  kube::test::get_object_assert 'deployments web' "{{${annotations_field}.annotatekey1}}" 'annotatevalue1'
  # Command
  output_message=$(kubectl annotate deployments annotatekey2=annotatevalue2 --all --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl annotate deployments annotatekey3=annotatevalue3 -l run=web 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl annotate deployments annotatekey4=annotatevalue4 -l run=web --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: DEPLOYMENT has annotation
  kube::test::if_has_string "${output_message}" 'deployment "web" annotated'
  kube::test::get_object_assert 'deployments web' "{{${annotations_field}.annotatekey4}}" 'annotatevalue4'
  # Command
  output_message=$(kubectl annotate deployments annotatekey5=annotatevalue5 -l run=web --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl annotate deployments annotatekey6=annotatevalue6 -l run=web --all --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: DEPLOYMENT has annotation
  kube::test::if_has_string "${output_message}" 'deployment "web" annotated'
  kube::test::get_object_assert 'deployments web' "{{${annotations_field}.annotatekey6}}" 'annotatevalue6'
  # Command
  output_message=$(kubectl annotate deployments web annotatekey7=annotatevalue7 2>&1 "${kube_flags[@]}")
  # Post-condition: web DEPLOYMENT has annotation
  kube::test::if_has_string "${output_message}" 'deployment "web" annotated'
  kube::test::get_object_assert 'deployments web' "{{${annotations_field}.annotatekey7}}" 'annotatevalue7'

  ### Test kubectl edit --include-uninitialized
  [ "$(EDITOR=cat kubectl edit deployments 2>&1 "${kube_flags[@]}" | grep 'edit cancelled, no objects found')" ]
  [ "$(EDITOR=cat kubectl edit deployments --include-uninitialized 2>&1 "${kube_flags[@]}" | grep 'Edit cancelled, no changes made.')" ]

  ### Test kubectl set image --include-uninitialized
  # Command
  output_message=$(kubectl set image deployments *=nginx:1.11 --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "image updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'image updated'
  # Command
  output_message=$(kubectl set image deployments *=nginx:1.11 --all --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl set image deployments *=nginx:1.11 -l run=web 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl set image deployments *=nginx:1.12 -l run=web --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "image updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'image updated'
  # Command
  output_message=$(kubectl set image deployments *=nginx:1.13 -l run=web --include-uninitialized --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "image updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'image updated'

  ### Test kubectl set resources --include-uninitialized
  # Command
  output_message=$(kubectl set resources deployments --limits=cpu=200m,memory=512Mi --requests=cpu=100m,memory=256Mi --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "resource requirements updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'resource requirements updated'
  # Command
  output_message=$(kubectl set resources deployments --limits=cpu=200m,memory=512Mi --requests=cpu=100m,memory=256Mi --all --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl set resources deployments --limits=cpu=200m,memory=512Mi --requests=cpu=100m,memory=256Mi -l run=web 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl set resources deployments --limits=cpu=200m,memory=512Mi --requests=cpu=200m,memory=256Mi -l run=web --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "resource requirements updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'resource requirements updated'
  # Command
  output_message=$(kubectl set resources deployments --limits=cpu=200m,memory=512Mi --requests=cpu=100m,memory=512Mi -l run=web --include-uninitialized --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "resource requirements updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'resource requirements updated'

  ### Test kubectl set selector --include-uninitialized
  # Create a service with initializer
  kubectl create --request-timeout=1 -f hack/testdata/initializer-redis-master-service.yaml 2>&1 "${kube_flags[@]}" || true
  # Command
  output_message=$(kubectl set selector services role=padawan --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "selector updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'selector updated'
  # Command
  output_message=$(kubectl set selector services role=padawan --all --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"

  ### Test kubectl set subject --include-uninitialized
  # Create a create clusterrolebinding with initializer
  kubectl create --request-timeout=1 -f hack/testdata/initializer-clusterrolebinding.yaml 2>&1 "${kube_flags[@]}" || true
  kube::test::get_object_assert clusterrolebinding/super-admin "{{range.subjects}}{{.name}}:{{end}}" 'super-admin:'
  # Command
  output_message=$(kubectl set subject clusterrolebinding --user=foo --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "subjects updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'subjects updated'
  # Command
  output_message=$(kubectl set subject clusterrolebinding --user=foo --all --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl set subject clusterrolebinding --user=foo -l clusterrolebinding=super 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl set subject clusterrolebinding --user=foo -l clusterrolebinding=super --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "subjects updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'subjects updated'
  # Command
  output_message=$(kubectl set subject clusterrolebinding --user=foo -l clusterrolebinding=super --include-uninitialized --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "subjects updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'subjects updated'

  ### Test kubectl set serviceaccount --include-uninitialized
  # Command
  output_message=$(kubectl set serviceaccount deployment serviceaccount1 --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "serviceaccount updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'serviceaccount updated'
  # Command
  output_message=$(kubectl set serviceaccount deployment serviceaccount1 --all --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"

  ### Test kubectl delete --include-uninitialized
  kube::test::get_object_assert clusterrolebinding/super-admin "{{range.subjects}}{{.name}}:{{end}}" 'super-admin:'
  # Command
  output_message=$(kubectl delete clusterrolebinding --all --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "No resources found" should be part of the output
  kube::test::if_has_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl delete clusterrolebinding --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "deleted" should be part of the output
  kube::test::if_has_string "${output_message}" 'deleted'
  kube::test::get_object_assert clusterrolebinding/super-admin "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Test kubectl apply --include-uninitialized
  # Pre-Condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # apply pod a
  kubectl apply --prune --request-timeout=20 --include-uninitialized=false --all -f hack/testdata/prune/a.yaml "${kube_flags[@]}" 2>&1
  # check right pod exists
  kube::test::get_object_assert pods/a "{{${id_field}}}" 'a'
  # Post-condition: Other uninitialized resources should not be pruned
  kube::test::get_object_assert deployments "{{range.items}}{{$id_field}}:{{end}}" 'web'
  kube::test::get_object_assert services/redis-master "{{range.items}}{{$id_field}}:{{end}}" 'redis-master'
  # cleanup
  kubectl delete pod a
  # apply pod a and prune uninitialized deployments web
  kubectl apply --prune --request-timeout=20 --all -f hack/testdata/prune/a.yaml "${kube_flags[@]}" 2>&1
  # check right pod exists
  kube::test::get_object_assert pods/a "{{${id_field}}}" 'a'
  # Post-condition: Other uninitialized resources should not be pruned
  kube::test::get_object_assert deployments/web "{{range.items}}{{$id_field}}:{{end}}" 'web'
  kube::test::get_object_assert services/redis-master "{{range.items}}{{$id_field}}:{{end}}" 'redis-master'
  # cleanup
  kubectl delete pod a
  # apply pod a and prune uninitialized deployments web
  kubectl apply --prune --request-timeout=20 --include-uninitialized --all -f hack/testdata/prune/a.yaml "${kube_flags[@]}" 2>&1
  # check right pod exists
  kube::test::get_object_assert pods/a "{{${id_field}}}" 'a'
  # Post-condition: Other uninitialized resources should not be pruned
  kube::test::get_object_assert deployments/web "{{range.items}}{{$id_field}}:{{end}}" 'web'
  kube::test::get_object_assert services/redis-master "{{range.items}}{{$id_field}}:{{end}}" 'redis-master'
  # cleanup
  kubectl delete pod a
  kubectl delete --request-timeout=1 deploy web
  kubectl delete --request-timeout=1 service redis-master

  set +o nounset
  set +o errexit
}
