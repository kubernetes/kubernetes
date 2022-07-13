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

run_daemonset_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl(v1:daemonsets)"

  ### Create a rolling update DaemonSet
  # Pre-condition: no DaemonSet exists
  kube::test::get_object_assert daemonsets "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command
  kubectl apply -f hack/testdata/rollingupdate-daemonset.yaml "${kube_flags[@]:?}"
  # Template Generation should be 1
  kube::test::get_object_assert 'daemonsets bind' "{{${generation_field:?}}}" '1'
  kubectl apply -f hack/testdata/rollingupdate-daemonset.yaml "${kube_flags[@]:?}"
  # Template Generation should stay 1
  kube::test::get_object_assert 'daemonsets bind' "{{${generation_field:?}}}" '1'
  # Test set commands
  kubectl set image daemonsets/bind "${kube_flags[@]:?}" "*=registry.k8s.io/pause:test-cmd"
  kube::test::get_object_assert 'daemonsets bind' "{{${generation_field:?}}}" '2'
  kubectl set env daemonsets/bind "${kube_flags[@]:?}" foo=bar
  kube::test::get_object_assert 'daemonsets bind' "{{${generation_field:?}}}" '3'
  kubectl set resources daemonsets/bind "${kube_flags[@]:?}" --limits=cpu=200m,memory=512Mi
  kube::test::get_object_assert 'daemonsets bind' "{{${generation_field:?}}}" '4'
  # pod has field for kubectl set field manager
  output_message=$(kubectl get daemonsets bind --show-managed-fields -o=jsonpath='{.metadata.managedFields[*].manager}' "${kube_flags[@]:?}" 2>&1)
  kube::test::if_has_string "${output_message}" 'kubectl-set'
  # Describe command should respect the chunk size parameter
  kube::test::describe_resource_chunk_size_assert daemonsets pods,events

  # Rollout restart should change generation
  kubectl rollout restart daemonset/bind "${kube_flags[@]:?}"
  kube::test::get_object_assert 'daemonsets bind' "{{${generation_field:?}}}" '5'

  # Clean up
  kubectl delete -f hack/testdata/rollingupdate-daemonset.yaml "${kube_flags[@]:?}"

  set +o nounset
  set +o errexit
}

run_daemonset_history_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl(v1:daemonsets, v1:controllerrevisions)"

  ### Test rolling back a DaemonSet
  # Pre-condition: no DaemonSet or its pods exists
  kube::test::get_object_assert daemonsets "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command
  # Create a DaemonSet (revision 1)
  kubectl apply -f hack/testdata/rollingupdate-daemonset.yaml --record "${kube_flags[@]:?}"
  kube::test::wait_object_assert controllerrevisions "{{range.items}}{{${annotations_field:?}}}:{{end}}" ".*rollingupdate-daemonset.yaml --record.*"
  # Rollback to revision 1 - should be no-op
  kubectl rollout undo daemonset --to-revision=1 "${kube_flags[@]:?}"
  kube::test::get_object_assert daemonset "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_PAUSE_V2}:"
  kube::test::get_object_assert daemonset "{{range.items}}{{${container_len:?}}}{{end}}" "1"
  # Update the DaemonSet (revision 2)
  kubectl apply -f hack/testdata/rollingupdate-daemonset-rv2.yaml --record "${kube_flags[@]:?}"
  kube::test::wait_object_assert daemonset "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DAEMONSET_R2}:"
  kube::test::wait_object_assert daemonset "{{range.items}}{{${image_field1:?}}}:{{end}}" "${IMAGE_DAEMONSET_R2_2}:"
  kube::test::get_object_assert daemonset "{{range.items}}{{${container_len:?}}}{{end}}" "2"
  kube::test::wait_object_assert controllerrevisions "{{range.items}}{{${annotations_field:?}}}:{{end}}" ".*rollingupdate-daemonset-rv2.yaml --record.*"
  # Get rollout history
  output_message=$(kubectl rollout history daemonset)
  kube::test::if_has_string "${output_message}" "daemonset.apps/bind"
  kube::test::if_has_string "${output_message}" "REVISION  CHANGE-CAUSE"
  kube::test::if_has_string "${output_message}" "1         kubectl apply"
  kube::test::if_has_string "${output_message}" "2         kubectl apply"
  # Get rollout history for a single revision
  output_message=$(kubectl rollout history daemonset --revision=1)
  kube::test::if_has_string "${output_message}" "daemonset.apps/bind with revision #1"
  kube::test::if_has_string "${output_message}" "Pod Template:"
  kube::test::if_has_string "${output_message}" "${IMAGE_PAUSE_V2}"
  # Get rollout history for a different single revision
  output_message=$(kubectl rollout history daemonset --revision=2)
  kube::test::if_has_string "${output_message}" "daemonset.apps/bind with revision #2"
  kube::test::if_has_string "${output_message}" "Pod Template:"
  kube::test::if_has_string "${output_message}" "${IMAGE_DAEMONSET_R2}"
  kube::test::if_has_string "${output_message}" "${IMAGE_DAEMONSET_R2_2}"
  # Rollback to revision 1 with dry-run - should be no-op
  kubectl rollout undo daemonset --dry-run=client "${kube_flags[@]:?}"
  kubectl rollout undo daemonset --dry-run=server "${kube_flags[@]:?}"
  kube::test::get_object_assert daemonset "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DAEMONSET_R2}:"
  kube::test::get_object_assert daemonset "{{range.items}}{{${image_field1:?}}}:{{end}}" "${IMAGE_DAEMONSET_R2_2}:"
  kube::test::get_object_assert daemonset "{{range.items}}{{${container_len:?}}}{{end}}" "2"
  # Rollback to revision 1
  kubectl rollout undo daemonset --to-revision=1 "${kube_flags[@]:?}"
  kube::test::wait_object_assert daemonset "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_PAUSE_V2}:"
  kube::test::get_object_assert daemonset "{{range.items}}{{${container_len:?}}}{{end}}" "1"
  # Get rollout history
  output_message=$(kubectl rollout history daemonset)
  kube::test::if_has_string "${output_message}" "daemonset.apps/bind"
  kube::test::if_has_string "${output_message}" "REVISION  CHANGE-CAUSE"
  kube::test::if_has_string "${output_message}" "2         kubectl apply"
  kube::test::if_has_string "${output_message}" "3         kubectl apply"
  # Rollback to revision 1000000 - should fail
  output_message=$(! kubectl rollout undo daemonset --to-revision=1000000 "${kube_flags[@]:?}" 2>&1)
  kube::test::if_has_string "${output_message}" "unable to find specified revision"
  kube::test::get_object_assert daemonset "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_PAUSE_V2}:"
  kube::test::get_object_assert daemonset "{{range.items}}{{${container_len:?}}}{{end}}" "1"
  # Rollback to last revision
  kubectl rollout undo daemonset "${kube_flags[@]:?}"
  kube::test::wait_object_assert daemonset "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DAEMONSET_R2}:"
  kube::test::wait_object_assert daemonset "{{range.items}}{{${image_field1:?}}}:{{end}}" "${IMAGE_DAEMONSET_R2_2}:"
  kube::test::get_object_assert daemonset "{{range.items}}{{${container_len:?}}}{{end}}" "2"
  # Get rollout history
  output_message=$(kubectl rollout history daemonset)
  kube::test::if_has_string "${output_message}" "daemonset.apps/bind"
  kube::test::if_has_string "${output_message}" "REVISION  CHANGE-CAUSE"
  kube::test::if_has_string "${output_message}" "3         kubectl apply"
  kube::test::if_has_string "${output_message}" "4         kubectl apply"
  # Clean up
  kubectl delete -f hack/testdata/rollingupdate-daemonset.yaml "${kube_flags[@]:?}"

  set +o nounset
  set +o errexit
}

run_kubectl_apply_deployments_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl apply deployments"
  ## kubectl apply should propagate user defined null values
  # Pre-Condition: no Deployments, ReplicaSets, Pods exist
  kube::test::get_object_assert deployments "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  kube::test::get_object_assert replicasets "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # apply base deployment
  kubectl apply -f hack/testdata/null-propagation/deployment-l1.yaml "${kube_flags[@]:?}"
  # check right deployment exists
  kube::test::get_object_assert 'deployments my-depl' "{{${id_field:?}}}" 'my-depl'
  # check right labels exists
  kube::test::get_object_assert 'deployments my-depl' "{{.spec.template.metadata.labels.l1}}" 'l1'
  kube::test::get_object_assert 'deployments my-depl' "{{.spec.selector.matchLabels.l1}}" 'l1'
  kube::test::get_object_assert 'deployments my-depl' "{{.metadata.labels.l1}}" 'l1'

  # apply new deployment with new template labels
  kubectl apply -f hack/testdata/null-propagation/deployment-l2.yaml "${kube_flags[@]:?}"
  # check right labels exists
  kube::test::get_object_assert 'deployments my-depl' "{{.spec.template.metadata.labels.l1}}" 'l1'
  kube::test::get_object_assert 'deployments my-depl' "{{.spec.selector.matchLabels.l1}}" 'l1'
  kube::test::get_object_assert 'deployments my-depl' "{{.metadata.labels.l1}}" '<no value>'

  # cleanup
  # need to explicitly remove replicasets and pods because we changed the deployment selector and orphaned things
  kubectl delete deployments,rs,pods --all --cascade=orphan --grace-period=0
  # Post-Condition: no Deployments, ReplicaSets, Pods exist
  kube::test::wait_object_assert deployments "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  kube::test::wait_object_assert replicasets "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''

  # kubectl apply deployment --overwrite=true --force=true
  # Pre-Condition: no deployment exists
  kube::test::get_object_assert deployments "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # apply deployment nginx
  kubectl apply -f hack/testdata/deployment-label-change1.yaml "${kube_flags[@]:?}"
  # check right deployment exists
  kube::test::get_object_assert 'deployment nginx' "{{${id_field:?}}}" 'nginx'
  # apply deployment with new labels and a conflicting resourceVersion
  output_message=$(! kubectl apply -f hack/testdata/deployment-label-change2.yaml 2>&1 "${kube_flags[@]:?}")
  kube::test::if_has_string "${output_message}" 'Error from server (Conflict)'
  # apply deployment with --force and --overwrite will succeed
  kubectl apply -f hack/testdata/deployment-label-change2.yaml --overwrite=true  --force=true --grace-period=10
  # check the changed deployment
  output_message=$(kubectl apply view-last-applied deploy/nginx -o json 2>&1 "${kube_flags[@]:?}" |grep nginx2)
  kube::test::if_has_string "${output_message}" '"name": "nginx2"'
  # applying a resource (with --force) that is both conflicting and invalid will
  # cause the server to only return a "Conflict" error when we attempt to patch.
  # This means that we will delete the existing resource after receiving 5 conflict
  # errors in a row from the server, and will attempt to create the modified
  # resource that we are passing to "apply". Since the modified resource is also
  # invalid, we will receive an invalid error when we attempt to create it, after
  # having deleted the old resource. Ensure that when this case is reached, the
  # old resource is restored once again, and the validation error is printed.
  output_message=$(! kubectl apply -f hack/testdata/deployment-label-change3.yaml --force 2>&1 "${kube_flags[@]:?}")
  kube::test::if_has_string "${output_message}" 'Invalid value'
  # Ensure that the old object has been restored
  kube::test::get_object_assert 'deployment nginx' "{{${template_labels:?}}}" 'nginx2'
  # cleanup
  kubectl delete deployments --all --grace-period=10

  set +o nounset
  set +o errexit
}

run_deployment_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing deployments"
  # Test kubectl create deployment (using default - old generator)
  kubectl create deployment test-nginx-extensions --image=registry.k8s.io/nginx:test-cmd
  # Post-Condition: Deployment "nginx" is created.
  kube::test::get_object_assert 'deploy test-nginx-extensions' "{{${container_name_field:?}}}" 'nginx'
  # and old generator was used, iow. old defaults are applied
  output_message=$(kubectl get deployment.apps/test-nginx-extensions -o jsonpath='{.spec.revisionHistoryLimit}')
  kube::test::if_has_not_string "${output_message}" '2'
  # Ensure we can interact with deployments through apps endpoints
  output_message=$(kubectl get deployment.apps -o=jsonpath='{.items[0].apiVersion}' 2>&1 "${kube_flags[@]:?}")
  kube::test::if_has_string "${output_message}" 'apps/v1'
  # Clean up
  kubectl delete deployment test-nginx-extensions "${kube_flags[@]:?}"

  # Test kubectl create deployment
  kubectl create deployment test-nginx-apps --image=registry.k8s.io/nginx:test-cmd
  # Post-Condition: Deployment "nginx" is created.
  kube::test::get_object_assert 'deploy test-nginx-apps' "{{${container_name_field:?}}}" 'nginx'
  # and new generator was used, iow. new defaults are applied
  output_message=$(kubectl get deployment/test-nginx-apps -o jsonpath='{.spec.revisionHistoryLimit}')
  kube::test::if_has_string "${output_message}" '10'
  # Ensure we can interact with deployments through apps endpoints
  output_message=$(kubectl get deployment.apps -o=jsonpath='{.items[0].apiVersion}' 2>&1 "${kube_flags[@]:?}")
  kube::test::if_has_string "${output_message}" 'apps/v1'
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert rs "Name:" "Pod Template:" "Labels:" "Selector:" "Controlled By" "Replicas:" "Pods Status:" "Volumes:"
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert pods "Name:" "Image:" "Node:" "Labels:" "Status:" "Controlled By"
  # Describe command should respect the chunk size parameter
  kube::test::describe_resource_chunk_size_assert deployments replicasets,events
  # Clean up
  kubectl delete deployment test-nginx-apps "${kube_flags[@]:?}"

  ### Test kubectl create deployment with image and command
  # Pre-Condition: No deployment exists.
  kube::test::get_object_assert deployment "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Dry-run command
  kubectl create deployment nginx-with-command --dry-run=client --image=registry.k8s.io/nginx:test-cmd -- /bin/sleep infinity
  kubectl create deployment nginx-with-command --dry-run=server --image=registry.k8s.io/nginx:test-cmd -- /bin/sleep infinity
  kube::test::get_object_assert deployment "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command
  kubectl create deployment nginx-with-command --image=registry.k8s.io/nginx:test-cmd -- /bin/sleep infinity
  # Post-Condition: Deployment "nginx" is created.
  kube::test::get_object_assert 'deploy nginx-with-command' "{{${container_name_field:?}}}" 'nginx'
  # Clean up
  kubectl delete deployment nginx-with-command "${kube_flags[@]:?}"

  ### Test kubectl create deployment should not fail validation
  # Pre-Condition: No deployment exists.
  kube::test::get_object_assert deployment "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/deployment-with-UnixUserID.yaml "${kube_flags[@]:?}"
  # Post-Condition: Deployment "deployment-with-unixuserid" is created.
  kube::test::get_object_assert deployment "{{range.items}}{{${id_field:?}}}:{{end}}" 'deployment-with-unixuserid:'
  # Clean up
  kubectl delete deployment deployment-with-unixuserid "${kube_flags[@]:?}"

  ### Test cascading deletion
  ## Test that rs is deleted when deployment is deleted.
  # Pre-condition: no deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Create deployment
  kubectl create -f test/fixtures/doc-yaml/user-guide/deployment.yaml "${kube_flags[@]:?}"
  # Wait for rs to come up.
  kube::test::wait_object_assert rs "{{range.items}}{{${rs_replicas_field:?}}}{{end}}" '3'
  # Deleting the deployment should delete the rs.
  # using empty value in cascade flag to make sure the backward compatibility.
  kubectl delete deployment nginx-deployment "${kube_flags[@]:?}" --cascade
  kube::test::wait_object_assert rs "{{range.items}}{{${id_field:?}}}:{{end}}" ''

  ## Test that rs is not deleted when deployment is deleted with cascading strategy set to orphan.
  # Pre-condition: no deployment and rs exist
  kube::test::get_object_assert deployment "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  kube::test::get_object_assert rs "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Create deployment
  kubectl create deployment nginx-deployment --image=registry.k8s.io/nginx:test-cmd
  # Wait for rs to come up.
  kube::test::wait_object_assert rs "{{range.items}}{{${rs_replicas_field:?}}}{{end}}" '1'
  # Delete the deployment with cascading strategy set to orphan.
  kubectl delete deployment nginx-deployment "${kube_flags[@]:?}" --cascade=orphan
  # Wait for the deployment to be deleted and then verify that rs is not
  # deleted.
  kube::test::wait_object_assert deployment "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  kube::test::get_object_assert rs "{{range.items}}{{${rs_replicas_field:?}}}{{end}}" '1'
  # Cleanup
  # Find the name of the rs to be deleted.
  output_message=$(kubectl get rs "${kube_flags[@]:?}" -o template --template="{{range.items}}{{${id_field:?}}}{{end}}")
  kubectl delete rs "${output_message}" "${kube_flags[@]:?}"

  ### Auto scale deployment
  # Pre-condition: no deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Pre-condition: no hpa exists
  kube::test::get_object_assert 'hpa' "{{range.items}}{{ if eq $id_field \\\"nginx-deployment\\\" }}found{{end}}{{end}}:" ':'
  # Command
  kubectl create -f test/fixtures/doc-yaml/user-guide/deployment.yaml "${kube_flags[@]:?}"
  kube::test::get_object_assert deployment "{{range.items}}{{${id_field:?}}}:{{end}}" 'nginx-deployment:'
  # Dry-run autoscale
  kubectl-with-retry autoscale deployment nginx-deployment --dry-run=client "${kube_flags[@]:?}" --min=2 --max=3
  kubectl-with-retry autoscale deployment nginx-deployment --dry-run=server "${kube_flags[@]:?}" --min=2 --max=3
  kube::test::get_object_assert 'hpa' "{{range.items}}{{ if eq $id_field \\\"nginx-deployment\\\" }}found{{end}}{{end}}:" ':'
  # autoscale 2~3 pods, no CPU utilization specified
  kubectl-with-retry autoscale deployment nginx-deployment "${kube_flags[@]:?}" --min=2 --max=3
  kube::test::get_object_assert 'hpa nginx-deployment' "{{${hpa_min_field:?}}} {{${hpa_max_field:?}}} {{${hpa_cpu_field:?}}}" '2 3 80'
  # Describe command should respect the chunk size parameter
  kube::test::describe_resource_chunk_size_assert horizontalpodautoscalers events
  # Clean up
  # Note that we should delete hpa first, otherwise it may fight with the deployment reaper.
  kubectl delete hpa nginx-deployment "${kube_flags[@]:?}"
  kubectl delete deployment.apps nginx-deployment "${kube_flags[@]:?}"

  ### Rollback a deployment
  # Pre-condition: no deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command
  # Create a deployment (revision 1)
  kubectl create -f hack/testdata/deployment-revision1.yaml "${kube_flags[@]:?}"
  kube::test::get_object_assert deployment "{{range.items}}{{${id_field:?}}}:{{end}}" 'nginx:'
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Rollback to revision 1 - should be no-op
  kubectl rollout undo deployment nginx --to-revision=1 "${kube_flags[@]:?}"
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Update the deployment (revision 2)
  kubectl apply -f hack/testdata/deployment-revision2.yaml "${kube_flags[@]:?}"
  kube::test::get_object_assert deployment.apps "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R2}:"
  # Rollback to revision 1 with dry-run - should be no-op
  kubectl rollout undo deployment nginx --dry-run=client "${kube_flags[@]:?}" | grep "test-cmd"
  kubectl rollout undo deployment nginx --dry-run=server "${kube_flags[@]:?}"
  kube::test::get_object_assert deployment.apps "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R2}:"
  # Rollback to revision 1
  kubectl rollout undo deployment nginx --to-revision=1 "${kube_flags[@]:?}"
  sleep 1
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Rollback to revision 1000000 - should be no-op
  ! kubectl rollout undo deployment nginx --to-revision=1000000 "${kube_flags[@]:?}" || exit 1
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Rollback to last revision
  kubectl rollout undo deployment nginx "${kube_flags[@]:?}"
  sleep 1
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R2}:"
  # Pause the deployment
  kubectl-with-retry rollout pause deployment nginx "${kube_flags[@]:?}"
  # A paused deployment cannot be rolled back
  ! kubectl rollout undo deployment nginx "${kube_flags[@]:?}" || exit 1
  # A paused deployment cannot be restarted
  ! kubectl rollout restart deployment nginx "${kube_flags[@]:?}" || exit 1
  # Resume the deployment
  kubectl-with-retry rollout resume deployment nginx "${kube_flags[@]:?}"
  # The resumed deployment can now be rolled back
  kubectl rollout undo deployment nginx "${kube_flags[@]:?}"
  # Check that the new replica set has all old revisions stored in an annotation
  newrs="$(kubectl describe deployment nginx | grep NewReplicaSet | awk '{print $2}')"
  kubectl get rs "${newrs}" -o yaml | grep "deployment.kubernetes.io/revision-history: 1,3"
  # Check that trying to watch the status of a superseded revision returns an error
  ! kubectl rollout status deployment/nginx --revision=3 || exit 1
  # Restarting the deployment creates a new replicaset
  kubectl rollout restart deployment/nginx
  sleep 1
  newrs="$(kubectl describe deployment nginx | grep NewReplicaSet | awk '{print $2}')"
  rs="$(kubectl get rs "${newrs}" -o yaml)"
  kube::test::if_has_string "${rs}" "deployment.kubernetes.io/revision: \"6\""
  # Deployment has field for kubectl rollout field manager
  output_message=$(kubectl get deployment nginx --show-managed-fields -o=jsonpath='{.metadata.managedFields[*].manager}' "${kube_flags[@]:?}" 2>&1)
  kube::test::if_has_string "${output_message}" 'kubectl-rollout'
  # Create second deployment
  ${SED} "s/name: nginx$/name: nginx2/" hack/testdata/deployment-revision1.yaml | kubectl create -f - "${kube_flags[@]:?}"
  # Deletion of both deployments should not be blocked
  kubectl delete deployment nginx2 "${kube_flags[@]:?}"
  # Clean up
  kubectl delete deployment nginx "${kube_flags[@]:?}"

  ### Set image of a deployment
  # Pre-condition: no deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Create a deployment
  kubectl create -f hack/testdata/deployment-multicontainer.yaml "${kube_flags[@]:?}"
  kube::test::get_object_assert deployment "{{range.items}}{{${id_field:?}}}:{{end}}" 'nginx-deployment:'
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field1:?}}}:{{end}}" "${IMAGE_PERL}:"
  # Dry-run set the deployment's image
  kubectl set image deployment nginx-deployment nginx="${IMAGE_DEPLOYMENT_R2}" --dry-run=client "${kube_flags[@]:?}"
  kubectl set image deployment nginx-deployment nginx="${IMAGE_DEPLOYMENT_R2}" --dry-run=server "${kube_flags[@]:?}"
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field1:?}}}:{{end}}" "${IMAGE_PERL}:"
  # Set the deployment's image
  kubectl set image deployment nginx-deployment nginx="${IMAGE_DEPLOYMENT_R2}" "${kube_flags[@]:?}"
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R2}:"
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field1:?}}}:{{end}}" "${IMAGE_PERL}:"
  # Set non-existing container should fail
  ! kubectl set image deployment nginx-deployment redis=redis "${kube_flags[@]:?}" || exit 1
  # Set image of deployments without specifying name
  kubectl set image deployments --all nginx="${IMAGE_DEPLOYMENT_R1}" "${kube_flags[@]:?}"
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field1:?}}}:{{end}}" "${IMAGE_PERL}:"
  # Set image of a deployment specified by file
  kubectl set image -f hack/testdata/deployment-multicontainer.yaml nginx="${IMAGE_DEPLOYMENT_R2}" "${kube_flags[@]:?}"
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R2}:"
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field1:?}}}:{{end}}" "${IMAGE_PERL}:"
  # Set image of a local file without talking to the server
  kubectl set image -f hack/testdata/deployment-multicontainer.yaml nginx="${IMAGE_DEPLOYMENT_R1}" "${kube_flags[@]:?}" --local -o yaml
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R2}:"
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field1:?}}}:{{end}}" "${IMAGE_PERL}:"
  # Set image of all containers of the deployment
  kubectl set image deployment nginx-deployment "*=${IMAGE_DEPLOYMENT_R1}" "${kube_flags[@]:?}"
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field1:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Set image of all containers of the deployment again when image not change
  kubectl set image deployment nginx-deployment "*=${IMAGE_DEPLOYMENT_R1}" "${kube_flags[@]:?}"
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field1:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Clean up
  kubectl delete deployment nginx-deployment "${kube_flags[@]:?}"

  ### Set env of a deployment
  # Pre-condition: no deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Create a deployment
  kubectl create -f hack/testdata/deployment-multicontainer.yaml "${kube_flags[@]:?}"
  kubectl create -f hack/testdata/configmap.yaml "${kube_flags[@]:?}"
  kubectl create -f hack/testdata/secret.yaml "${kube_flags[@]:?}"
  kube::test::get_object_assert deployment "{{range.items}}{{${id_field:?}}}:{{end}}" 'nginx-deployment:'
  #configmap is special here due to controller will create kube-root-ca.crt for each namespace automatically
  kube::test::get_object_assert 'configmaps/test-set-env-config' "{{${id_field:?}}}" 'test-set-env-config'
  kube::test::get_object_assert secret "{{range.items}}{{${id_field:?}}}:{{end}}" 'test-set-env-secret:'
  # Set env of deployments by configmap from keys
  kubectl set env deployment nginx-deployment --keys=key-2 --from=configmap/test-set-env-config "${kube_flags[@]:?}"
  # Assert correct value in deployment env
  kube::test::get_object_assert 'deploy nginx-deployment' "{{ (index (index .spec.template.spec.containers 0).env 0).name}}" 'KEY_2'
  # Assert single value in deployment env
  kube::test::get_object_assert 'deploy nginx-deployment' "{{ len (index .spec.template.spec.containers 0).env }}" '1'
  # Dry-run set env
  kubectl set env deployment nginx-deployment --dry-run=client --from=configmap/test-set-env-config "${kube_flags[@]:?}"
  kubectl set env deployment nginx-deployment --dry-run=server --from=configmap/test-set-env-config "${kube_flags[@]:?}"
  kube::test::get_object_assert 'deploy nginx-deployment' "{{ len (index .spec.template.spec.containers 0).env }}" '1'
  # Set env of deployments by configmap
  kubectl set env deployment nginx-deployment --from=configmap/test-set-env-config "${kube_flags[@]:?}"
  # Assert all values in deployment env
  kube::test::get_object_assert 'deploy nginx-deployment' "{{ len (index .spec.template.spec.containers 0).env }}" '2'
  # Set env of deployments for all container
  kubectl set env deployment nginx-deployment env=prod "${kube_flags[@]:?}"
  # Set env of deployments for specific container
  kubectl set env deployment nginx-deployment superenv=superprod -c=nginx "${kube_flags[@]:?}"
  # Set env of deployments by secret from keys
  kubectl set env deployment nginx-deployment --keys=username --from=secret/test-set-env-secret "${kube_flags[@]:?}"
  # Set env of deployments by secret
  kubectl set env deployment nginx-deployment --from=secret/test-set-env-secret "${kube_flags[@]:?}"
  # Remove specific env of deployment
  kubectl set env deployment nginx-deployment env-
  # Assert that we cannot use standard input for both resource and environment variable
  output_message="$(echo SOME_ENV_VAR_KEY=SOME_ENV_VAR_VAL | kubectl set env -f - - "${kube_flags[@]:?}" 2>&1 || true)"
  kube::test::if_has_string "${output_message}" 'standard input cannot be used for multiple arguments'
  # Clean up
  kubectl delete deployment nginx-deployment "${kube_flags[@]:?}"
  kubectl delete configmap test-set-env-config "${kube_flags[@]:?}"
  kubectl delete secret test-set-env-secret "${kube_flags[@]:?}"

  ### Get rollout history
  # Pre-condition: no deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Create a deployment
  kubectl create -f hack/testdata/deployment-multicontainer.yaml "${kube_flags[@]:?}"
  kube::test::get_object_assert deployment "{{range.items}}{{${id_field:?}}}:{{end}}" 'nginx-deployment:'
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field1:?}}}:{{end}}" "${IMAGE_PERL}:"
  # Set the deployment's image
  kubectl set image deployment nginx-deployment nginx="${IMAGE_DEPLOYMENT_R2}" "${kube_flags[@]:?}"
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R2}:"
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field1:?}}}:{{end}}" "${IMAGE_PERL}:"
  # Get rollout history
  output_message=$(kubectl rollout history deployment nginx-deployment)
  kube::test::if_has_string "${output_message}" "deployment.apps/nginx-deployment"
  kube::test::if_has_string "${output_message}" "REVISION  CHANGE-CAUSE"
  kube::test::if_has_string "${output_message}" "1         <none>"
  kube::test::if_has_string "${output_message}" "2         <none>"
  kube::test::if_has_not_string "${output_message}" "3         <none>"
  # Get rollout history for a single revision
  output_message=$(kubectl rollout history deployment nginx-deployment --revision=1)
  kube::test::if_has_string "${output_message}" "deployment.apps/nginx-deployment with revision #1"
  kube::test::if_has_string "${output_message}" "Pod Template:"
  kube::test::if_has_string "${output_message}" "${IMAGE_DEPLOYMENT_R1}"
  kube::test::if_has_string "${output_message}" "${IMAGE_PERL}"
  # Get rollout history for a different single revision
  output_message=$(kubectl rollout history deployment nginx-deployment --revision=2)
  kube::test::if_has_string "${output_message}" "deployment.apps/nginx-deployment with revision #2"
  kube::test::if_has_string "${output_message}" "Pod Template:"
  kube::test::if_has_string "${output_message}" "${IMAGE_DEPLOYMENT_R2}"
  kube::test::if_has_string "${output_message}" "${IMAGE_PERL}"
  # Clean up
  kubectl delete deployment nginx-deployment "${kube_flags[@]:?}"

  set +o nounset
  set +o errexit
}

run_statefulset_history_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl(v1:statefulsets, v1:controllerrevisions)"

  ### Test rolling back a StatefulSet
  # Pre-condition: no statefulset or its pods exists
  kube::test::get_object_assert statefulset "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command
  # Create a StatefulSet (revision 1)
  kubectl apply -f hack/testdata/rollingupdate-statefulset.yaml --record "${kube_flags[@]:?}"
  kube::test::wait_object_assert controllerrevisions "{{range.items}}{{${annotations_field:?}}}:{{end}}" ".*rollingupdate-statefulset.yaml --record.*"
  # Rollback to revision 1 - should be no-op
  kubectl rollout undo statefulset --to-revision=1 "${kube_flags[@]:?}"
  kube::test::get_object_assert statefulset "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_STATEFULSET_R1}:"
  kube::test::get_object_assert statefulset "{{range.items}}{{${container_len:?}}}{{end}}" "1"
  # Update the statefulset (revision 2)
  kubectl apply -f hack/testdata/rollingupdate-statefulset-rv2.yaml --record "${kube_flags[@]:?}"
  kube::test::wait_object_assert statefulset "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_STATEFULSET_R2}:"
  kube::test::wait_object_assert statefulset "{{range.items}}{{${image_field1:?}}}:{{end}}" "${IMAGE_PAUSE_V2}:"
  kube::test::get_object_assert statefulset "{{range.items}}{{${container_len:?}}}{{end}}" "2"
  kube::test::wait_object_assert controllerrevisions "{{range.items}}{{${annotations_field:?}}}:{{end}}" ".*rollingupdate-statefulset-rv2.yaml --record.*"
  # Get rollout history
  output_message=$(kubectl rollout history statefulset)
  kube::test::if_has_string "${output_message}" "statefulset.apps/nginx"
  kube::test::if_has_string "${output_message}" "REVISION  CHANGE-CAUSE"
  kube::test::if_has_string "${output_message}" "1         kubectl apply"
  kube::test::if_has_string "${output_message}" "2         kubectl apply"
  # Get rollout history for a single revision
  output_message=$(kubectl rollout history statefulset --revision=1)
  kube::test::if_has_string "${output_message}" "statefulset.apps/nginx with revision #1"
  kube::test::if_has_string "${output_message}" "Pod Template:"
  kube::test::if_has_string "${output_message}" "${IMAGE_STATEFULSET_R1}"
  # Get rollout history for a different single revision
  output_message=$(kubectl rollout history statefulset --revision=2)
  kube::test::if_has_string "${output_message}" "statefulset.apps/nginx with revision #2"
  kube::test::if_has_string "${output_message}" "Pod Template:"
  kube::test::if_has_string "${output_message}" "${IMAGE_STATEFULSET_R2}"
  kube::test::if_has_string "${output_message}" "${IMAGE_PAUSE_V2}"
  # Rollback to revision 1 with dry-run - should be no-op
  kubectl rollout undo statefulset --dry-run=client "${kube_flags[@]:?}"
  kubectl rollout undo statefulset --dry-run=server "${kube_flags[@]:?}"
  kube::test::get_object_assert statefulset "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_STATEFULSET_R2}:"
  kube::test::get_object_assert statefulset "{{range.items}}{{${image_field1:?}}}:{{end}}" "${IMAGE_PAUSE_V2}:"
  kube::test::get_object_assert statefulset "{{range.items}}{{${container_len:?}}}{{end}}" "2"
  # Rollback to revision 1
  kubectl rollout undo statefulset --to-revision=1 "${kube_flags[@]:?}"
  kube::test::wait_object_assert statefulset "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_STATEFULSET_R1}:"
  kube::test::get_object_assert statefulset "{{range.items}}{{${container_len:?}}}{{end}}" "1"
  # Get rollout history
  output_message=$(kubectl rollout history statefulset)
  kube::test::if_has_string "${output_message}" "statefulset.apps/nginx"
  kube::test::if_has_string "${output_message}" "REVISION  CHANGE-CAUSE"
  kube::test::if_has_string "${output_message}" "2         kubectl apply"
  kube::test::if_has_string "${output_message}" "3         kubectl apply"
  # Rollback to revision 1000000 - should fail
  output_message=$(! kubectl rollout undo statefulset --to-revision=1000000 "${kube_flags[@]:?}" 2>&1)
  kube::test::if_has_string "${output_message}" "unable to find specified revision"
  kube::test::get_object_assert statefulset "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_STATEFULSET_R1}:"
  kube::test::get_object_assert statefulset "{{range.items}}{{${container_len:?}}}{{end}}" "1"
  # Rollback to last revision
  kubectl rollout undo statefulset "${kube_flags[@]:?}"
  kube::test::wait_object_assert statefulset "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_STATEFULSET_R2}:"
  kube::test::wait_object_assert statefulset "{{range.items}}{{${image_field1:?}}}:{{end}}" "${IMAGE_PAUSE_V2}:"
  kube::test::get_object_assert statefulset "{{range.items}}{{${container_len:?}}}{{end}}" "2"
  # Get rollout history
  output_message=$(kubectl rollout history statefulset)
  kube::test::if_has_string "${output_message}" "statefulset.apps/nginx"
  kube::test::if_has_string "${output_message}" "REVISION  CHANGE-CAUSE"
  kube::test::if_has_string "${output_message}" "3         kubectl apply"
  kube::test::if_has_string "${output_message}" "4         kubectl apply"
  # Clean up - delete newest configuration
  kubectl delete -f hack/testdata/rollingupdate-statefulset-rv2.yaml "${kube_flags[@]:?}"
  # Post-condition: no pods from statefulset controller
  wait-for-pods-with-label "app=nginx-statefulset" ""

  set +o nounset
  set +o errexit
}

run_stateful_set_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl(v1:statefulsets)"

  ### Create and stop statefulset, make sure it doesn't leak pods
  # Pre-condition: no statefulset exists
  kube::test::get_object_assert statefulset "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command: create statefulset
  kubectl create -f hack/testdata/rollingupdate-statefulset.yaml "${kube_flags[@]:?}"

  # Describe command should respect the chunk size parameter
  kube::test::describe_resource_chunk_size_assert statefulsets pods,events

  ### Scale statefulset test with current-replicas and replicas
  # Pre-condition: 0 replicas
  kube::test::get_object_assert 'statefulset nginx' "{{${statefulset_replicas_field:?}}}" '0'
  kube::test::wait_object_assert 'statefulset nginx' "{{${statefulset_observed_generation:?}}}" '1'
  # Command: Scale up
  kubectl scale --current-replicas=0 --replicas=1 statefulset nginx "${kube_flags[@]:?}"
  # Post-condition: 1 replica, named nginx-0
  kube::test::get_object_assert 'statefulset nginx' "{{${statefulset_replicas_field:?}}}" '1'
  kube::test::wait_object_assert 'statefulset nginx' "{{${statefulset_observed_generation:?}}}" '2'
  # Typically we'd wait and confirm that N>1 replicas are up, but this framework
  # doesn't start  the scheduler, so pet-0 will block all others.
  # TODO: test robust scaling in an e2e.
  wait-for-pods-with-label "app=nginx-statefulset" "nginx-0"

  # Rollout restart should change generation
  kubectl rollout restart statefulset nginx "${kube_flags[@]}"
  kube::test::get_object_assert 'statefulset nginx' "{{$statefulset_observed_generation}}" '3'

  ### Clean up
  kubectl delete -f hack/testdata/rollingupdate-statefulset.yaml "${kube_flags[@]:?}"
  # Post-condition: no pods from statefulset controller
  wait-for-pods-with-label "app=nginx-statefulset" ""

  set +o nounset
  set +o errexit

}

run_rs_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl(v1:replicasets)"

  ### Create and stop a replica set, make sure it doesn't leak pods
  # Pre-condition: no replica set exists
  kube::test::get_object_assert rs "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-replicaset.yaml "${kube_flags[@]:?}"
  kube::log::status "Deleting rs"
  kubectl delete rs frontend "${kube_flags[@]:?}"
  # Post-condition: no pods from frontend replica set
  kube::test::wait_object_assert 'pods -l "tier=frontend"' "{{range.items}}{{${id_field:?}}}:{{end}}" ''

  ### Create and then delete a replica set with cascading strategy set to orphan, make sure it doesn't delete pods.
  # Pre-condition: no replica set exists
  kube::test::get_object_assert rs "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-replicaset.yaml "${kube_flags[@]}"
  # wait for all 3 pods to be set up
  kube::test::wait_object_assert 'pods -l "tier=frontend"' "{{range.items}}{{${pod_container_name_field:?}}}:{{end}}" 'php-redis:php-redis:php-redis:'
  kube::log::status "Deleting rs"
  kubectl delete rs frontend "${kube_flags[@]:?}" --cascade=orphan
  # Wait for the rs to be deleted.
  kube::test::wait_object_assert rs "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Post-condition: All 3 pods still remain from frontend replica set
  kube::test::get_object_assert 'pods -l "tier=frontend"' "{{range.items}}{{$pod_container_name_field}}:{{end}}" 'php-redis:php-redis:php-redis:'
  # Cleanup
  kubectl delete pods -l "tier=frontend" "${kube_flags[@]:?}"
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''

  ### Create replica set frontend from YAML
  # Pre-condition: no replica set exists
  kube::test::get_object_assert rs "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-replicaset.yaml "${kube_flags[@]:?}"
  # Post-condition: frontend replica set is created
  kube::test::get_object_assert rs "{{range.items}}{{${id_field:?}}}:{{end}}" 'frontend:'
  # Describe command should print detailed information
  kube::test::describe_object_assert rs 'frontend' "Name:" "Pod Template:" "Labels:" "Selector:" "Replicas:" "Pods Status:" "Volumes:"
  # Describe command should print events information by default
  kube::test::describe_object_events_assert rs 'frontend'
  # Describe command should not print events information when show-events=false
  kube::test::describe_object_events_assert rs 'frontend' false
  # Describe command should print events information when show-events=true
  kube::test::describe_object_events_assert rs 'frontend' true
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert rs "Name:" "Pod Template:" "Labels:" "Selector:" "Replicas:" "Pods Status:" "Volumes:"
  # Describe command should print events information by default
  kube::test::describe_resource_events_assert rs
  # Describe command should not print events information when show-events=false
  kube::test::describe_resource_events_assert rs false
  # Describe command should print events information when show-events=true
  kube::test::describe_resource_events_assert rs true
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert pods "Name:" "Image:" "Node:" "Labels:" "Status:" "Controlled By"
  # Describe command should respect the chunk size parameter
  kube::test::describe_resource_chunk_size_assert replicasets pods,events

  ### Scale replica set frontend with current-replicas and replicas
  # Pre-condition: 3 replicas
  kube::test::get_object_assert 'rs frontend' "{{${rs_replicas_field:?}}}" '3'
  # Dry-run Command
  kubectl scale --dry-run=client --current-replicas=3 --replicas=2 replicasets frontend "${kube_flags[@]:?}"
  kubectl scale --dry-run=server --current-replicas=3 --replicas=2 replicasets frontend "${kube_flags[@]:?}"
  kube::test::get_object_assert 'rs frontend' "{{${rs_replicas_field:?}}}" '3'
  # Command
  kubectl scale --current-replicas=3 --replicas=2 replicasets frontend "${kube_flags[@]:?}"
  # Post-condition: 2 replicas
  kube::test::get_object_assert 'rs frontend' "{{${rs_replicas_field:?}}}" '2'

  # Set up three deploy, two deploy have same label
  kubectl create -f hack/testdata/scale-deploy-1.yaml "${kube_flags[@]:?}"
  kubectl create -f hack/testdata/scale-deploy-2.yaml "${kube_flags[@]:?}"
  kubectl create -f hack/testdata/scale-deploy-3.yaml "${kube_flags[@]:?}"
  kube::test::get_object_assert 'deploy scale-1' "{{.spec.replicas}}" '1'
  kube::test::get_object_assert 'deploy scale-2' "{{.spec.replicas}}" '1'
  kube::test::get_object_assert 'deploy scale-3' "{{.spec.replicas}}" '1'
  # Test kubectl scale --all with dry run
  kubectl scale deploy --replicas=3 --all --dry-run=client
  kubectl scale deploy --replicas=3 --all --dry-run=server
  kube::test::get_object_assert 'deploy scale-1' "{{.spec.replicas}}" '1'
  kube::test::get_object_assert 'deploy scale-2' "{{.spec.replicas}}" '1'
  kube::test::get_object_assert 'deploy scale-3' "{{.spec.replicas}}" '1'
  # Test kubectl scale --selector
  kubectl scale deploy --replicas=2 -l run=hello
  kube::test::get_object_assert 'deploy scale-1' "{{.spec.replicas}}" '2'
  kube::test::get_object_assert 'deploy scale-2' "{{.spec.replicas}}" '2'
  kube::test::get_object_assert 'deploy scale-3' "{{.spec.replicas}}" '1'
  # Test kubectl scale --all
  kubectl scale deploy --replicas=3 --all
  kube::test::get_object_assert 'deploy scale-1' "{{.spec.replicas}}" '3'
  kube::test::get_object_assert 'deploy scale-2' "{{.spec.replicas}}" '3'
  kube::test::get_object_assert 'deploy scale-3' "{{.spec.replicas}}" '3'
  # Clean-up
  kubectl delete rs frontend "${kube_flags[@]:?}"
  kubectl delete deploy scale-1 scale-2 scale-3 "${kube_flags[@]:?}"

  ### Expose replica set as service
  kubectl create -f hack/testdata/frontend-replicaset.yaml "${kube_flags[@]:?}"
  # Pre-condition: 3 replicas
  kube::test::get_object_assert 'rs frontend' "{{${rs_replicas_field:?}}}" '3'
  # Command
  kubectl expose rs frontend --port=80 "${kube_flags[@]:?}"
  # Post-condition: service exists and the port is unnamed
  kube::test::get_object_assert 'service frontend' "{{${port_name:?}}} {{${port_field:?}}}" '<no value> 80'
  # Cleanup services
  kubectl delete service frontend "${kube_flags[@]:?}"

  # Test set commands
  # Pre-condition: frontend replica set exists at generation 1
  kube::test::get_object_assert 'rs frontend' "{{${generation_field:?}}}" '1'
  kubectl set image rs/frontend "${kube_flags[@]:?}" "*=registry.k8s.io/pause:test-cmd"
  kube::test::get_object_assert 'rs frontend' "{{${generation_field:?}}}" '2'
  kubectl set env rs/frontend "${kube_flags[@]:?}" foo=bar
  kube::test::get_object_assert 'rs frontend' "{{${generation_field:?}}}" '3'
  kubectl set resources rs/frontend --dry-run=client "${kube_flags[@]:?}" --limits=cpu=200m,memory=512Mi
  kubectl set resources rs/frontend --dry-run=server "${kube_flags[@]:?}" --limits=cpu=200m,memory=512Mi
  kube::test::get_object_assert 'rs frontend' "{{${generation_field:?}}}" '3'
  kubectl set resources rs/frontend "${kube_flags[@]:?}" --limits=cpu=200m,memory=512Mi
  kube::test::get_object_assert 'rs frontend' "{{${generation_field:?}}}" '4'
  kubectl set serviceaccount rs/frontend --dry-run=client "${kube_flags[@]:?}" serviceaccount1
  kubectl set serviceaccount rs/frontend --dry-run=server "${kube_flags[@]:?}" serviceaccount1
  kube::test::get_object_assert 'rs frontend' "{{${generation_field:?}}}" '4'
  kubectl set serviceaccount rs/frontend "${kube_flags[@]:?}" serviceaccount1
  kube::test::get_object_assert 'rs frontend' "{{${generation_field:?}}}" '5'

  # RS has field for kubectl set field manager
  output_message=$(kubectl get rs frontend --show-managed-fields -o=jsonpath='{.metadata.managedFields[*].manager}' "${kube_flags[@]:?}" 2>&1)
  kube::test::if_has_string "${output_message}" 'kubectl-set'

  ### Delete replica set with id
  # Pre-condition: frontend replica set exists
  kube::test::get_object_assert rs "{{range.items}}{{${id_field:?}}}:{{end}}" 'frontend:'
  # Command
  kubectl delete rs frontend "${kube_flags[@]:?}"
  # Post-condition: no replica set exists
  kube::test::get_object_assert rs "{{range.items}}{{${id_field:?}}}:{{end}}" ''

  ### Create two replica sets
  # Pre-condition: no replica set exists
  kube::test::get_object_assert rs "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-replicaset.yaml "${kube_flags[@]:?}"
  kubectl create -f hack/testdata/redis-slave-replicaset.yaml "${kube_flags[@]:?}"
  # Post-condition: frontend and redis-slave
  kube::test::get_object_assert rs "{{range.items}}{{${id_field:?}}}:{{end}}" 'frontend:redis-slave:'

  ### Delete multiple replica sets at once
  # Pre-condition: frontend and redis-slave
  kube::test::get_object_assert rs "{{range.items}}{{${id_field:?}}}:{{end}}" 'frontend:redis-slave:'
  # Command
  kubectl delete rs frontend redis-slave "${kube_flags[@]:?}" # delete multiple replica sets at once
  # Post-condition: no replica set exists
  kube::test::get_object_assert rs "{{range.items}}{{${id_field:?}}}:{{end}}" ''

  if kube::test::if_supports_resource "horizontalpodautoscalers" ; then
    ### Auto scale replica set
    # Pre-condition: no replica set exists
    kube::test::get_object_assert rs "{{range.items}}{{${id_field:?}}}:{{end}}" ''
    # Command
    kubectl create -f hack/testdata/frontend-replicaset.yaml "${kube_flags[@]:?}"
    kube::test::get_object_assert rs "{{range.items}}{{${id_field:?}}}:{{end}}" 'frontend:'
    # autoscale 1~2 pods, CPU utilization 70%, replica set specified by file
    kubectl autoscale -f hack/testdata/frontend-replicaset.yaml "${kube_flags[@]:?}" --max=2 --cpu-percent=70
    kube::test::get_object_assert 'hpa frontend' "{{${hpa_min_field:?}}} {{${hpa_max_field:?}}} {{${hpa_cpu_field:?}}}" '1 2 70'
    kubectl delete hpa frontend "${kube_flags[@]:?}"
    # autoscale 2~3 pods, no CPU utilization specified, replica set specified by name
    kubectl autoscale rs frontend "${kube_flags[@]:?}" --min=2 --max=3
    kube::test::get_object_assert 'hpa frontend' "{{${hpa_min_field:?}}} {{${hpa_max_field:?}}} {{${hpa_cpu_field:?}}}" '2 3 80'
    # HorizontalPodAutoscaler has field for kubectl autoscale field manager
    output_message=$(kubectl get hpa frontend -o=jsonpath='{.metadata.managedFields[*].manager}' "${kube_flags[@]:?}" 2>&1)
    kube::test::if_has_string "${output_message}" 'kubectl-autoscale'
    # Clean up
    kubectl delete hpa frontend "${kube_flags[@]:?}"
    # autoscale without specifying --max should fail
    ! kubectl autoscale rs frontend "${kube_flags[@]:?}" || exit 1
    # Clean up
    kubectl delete rs frontend "${kube_flags[@]:?}"
  fi

  set +o nounset
  set +o errexit
}
