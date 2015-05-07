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

# This command checks that the built commands can function together for
# simple scenarios.  It does not require Docker so it can run in travis.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/test.sh"

function cleanup()
{
    [[ -n ${APISERVER_PID-} ]] && kill ${APISERVER_PID} 1>&2 2>/dev/null
    [[ -n ${CTLRMGR_PID-} ]] && kill ${CTLRMGR_PID} 1>&2 2>/dev/null
    [[ -n ${KUBELET_PID-} ]] && kill ${KUBELET_PID} 1>&2 2>/dev/null
    [[ -n ${PROXY_PID-} ]] && kill ${PROXY_PID} 1>&2 2>/dev/null

    kube::etcd::cleanup
    rm -rf "${KUBE_TEMP}"

    kube::log::status "Clean up complete"
}

trap cleanup EXIT SIGINT

kube::util::ensure-temp-dir
kube::etcd::start

ETCD_HOST=${ETCD_HOST:-127.0.0.1}
ETCD_PORT=${ETCD_PORT:-4001}
API_PORT=${API_PORT:-8080}
API_HOST=${API_HOST:-127.0.0.1}
KUBELET_PORT=${KUBELET_PORT:-10250}
KUBELET_HEALTHZ_PORT=${KUBELET_HEALTHZ_PORT:-10248}
CTLRMGR_PORT=${CTLRMGR_PORT:-10252}

# Check kubectl
kube::log::status "Running kubectl with no options"
"${KUBE_OUTPUT_HOSTBIN}/kubectl"

kube::log::status "Starting kubelet in masterless mode"
"${KUBE_OUTPUT_HOSTBIN}/kubelet" \
  --really_crash_for_testing=true \
  --root_dir=/tmp/kubelet.$$ \
  --cert_dir="${TMPDIR:-/tmp/}" \
  --docker_endpoint="fake://" \
  --hostname_override="127.0.0.1" \
  --address="127.0.0.1" \
  --port="$KUBELET_PORT" \
  --healthz_port="${KUBELET_HEALTHZ_PORT}" 1>&2 &
KUBELET_PID=$!
kube::util::wait_for_url "http://127.0.0.1:${KUBELET_HEALTHZ_PORT}/healthz" "kubelet: " 0.2 25
kill ${KUBELET_PID} 1>&2 2>/dev/null

kube::log::status "Starting kubelet in masterful mode"
"${KUBE_OUTPUT_HOSTBIN}/kubelet" \
  --really_crash_for_testing=true \
  --root_dir=/tmp/kubelet.$$ \
  --cert_dir="${TMPDIR:-/tmp/}" \
  --docker_endpoint="fake://" \
  --hostname_override="127.0.0.1" \
  --address="127.0.0.1" \
  --api_servers="${API_HOST}:${API_PORT}" \
  --auth_path="${KUBE_ROOT}/hack/.test-cmd-auth" \
  --port="$KUBELET_PORT" \
  --healthz_port="${KUBELET_HEALTHZ_PORT}" 1>&2 &
KUBELET_PID=$!

kube::util::wait_for_url "http://127.0.0.1:${KUBELET_HEALTHZ_PORT}/healthz" "kubelet: " 0.2 25

# Start kube-apiserver
kube::log::status "Starting kube-apiserver"
"${KUBE_OUTPUT_HOSTBIN}/kube-apiserver" \
  --address="127.0.0.1" \
  --public_address_override="127.0.0.1" \
  --port="${API_PORT}" \
  --etcd_servers="http://${ETCD_HOST}:${ETCD_PORT}" \
  --public_address_override="127.0.0.1" \
  --kubelet_port=${KUBELET_PORT} \
  --runtime_config=api/v1beta3 \
  --cert_dir="${TMPDIR:-/tmp/}" \
  --portal_net="10.0.0.0/24" 1>&2 &
APISERVER_PID=$!

kube::util::wait_for_url "http://127.0.0.1:${API_PORT}/healthz" "apiserver: "

# Start controller manager
kube::log::status "Starting CONTROLLER-MANAGER"
"${KUBE_OUTPUT_HOSTBIN}/kube-controller-manager" \
  --machines="127.0.0.1" \
  --master="127.0.0.1:${API_PORT}" 1>&2 &
CTLRMGR_PID=$!

kube::util::wait_for_url "http://127.0.0.1:${CTLRMGR_PORT}/healthz" "controller-manager: "
kube::util::wait_for_url "http://127.0.0.1:${API_PORT}/api/v1beta3/nodes/127.0.0.1" "apiserver(nodes): " 0.2 25

# Expose kubectl directly for readability
PATH="${KUBE_OUTPUT_HOSTBIN}":$PATH

kube_api_versions=(
  ""
  v1beta1
  v1beta2
  v1beta3
)
for version in "${kube_api_versions[@]}"; do
  if [[ -z "${version}" ]]; then
    kube_flags=(
      -s "http://127.0.0.1:${API_PORT}"
      --match-server-version
    )
    [ "$(kubectl get minions -t '{{ .apiVersion }}' "${kube_flags[@]}")" == "v1beta3" ]
  else
    kube_flags=(
      -s "http://127.0.0.1:${API_PORT}"
      --match-server-version
      --api-version="${version}"
    )
    [ "$(kubectl get minions -t '{{ .apiVersion }}' "${kube_flags[@]}")" == "${version}" ]
  fi
  id_field=".metadata.name"
  labels_field=".metadata.labels"
  service_selector_field=".spec.selector"
  rc_replicas_field=".spec.replicas"
  rc_status_replicas_field=".status.replicas"
  rc_container_image_field=".spec.template.spec.containers"
  port_field="(index .spec.ports 0).port"
  if [ "${version}" = "v1beta1" ] || [ "${version}" = "v1beta2" ]; then
    id_field=".id"
    labels_field=".labels"
    service_selector_field=".selector"
    rc_replicas_field=".desiredState.replicas"
    rc_status_replicas_field=".currentState.replicas"
    rc_container_image_field=".desiredState.podTemplate.desiredState.manifest.containers"
    port_field=".port"
  fi

  # Passing no arguments to create is an error
  ! kubectl create

  ###########################
  # POD creation / deletion #
  ###########################

  kube::log::status "Testing kubectl(${version}:pods)"

  ### Create POD valid-pod from JSON
  # Pre-condition: no POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create "${kube_flags[@]}" -f examples/limitrange/valid-pod.json
  # Post-condition: valid-pod POD is running
  kubectl get "${kube_flags[@]}" pods -o json
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  kube::test::get_object_assert 'pod valid-pod' "{{$id_field}}" 'valid-pod'
  kube::test::get_object_assert 'pod/valid-pod' "{{$id_field}}" 'valid-pod'
  kube::test::get_object_assert 'pods/valid-pod' "{{$id_field}}" 'valid-pod'
  # Describe command should print detailed information
  kube::test::describe_object_assert pods 'valid-pod' "Name:" "Image(s):" "Host:" "Labels:" "Status:" "Replication Controllers"

  ### Dump current valid-pod POD
  output_pod=$(kubectl get pod valid-pod -o yaml --output-version=v1beta3 "${kube_flags[@]}")

  ### Delete POD valid-pod by id
  # Pre-condition: valid-pod POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete pod valid-pod "${kube_flags[@]}"
  # Post-condition: no POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create POD valid-pod from dumped YAML
  # Pre-condition: no POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  echo "${output_pod}" | kubectl create -f - "${kube_flags[@]}"
  # Post-condition: valid-pod POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Delete POD valid-pod from JSON
  # Pre-condition: valid-pod POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete -f examples/limitrange/valid-pod.json "${kube_flags[@]}"
  # Post-condition: no POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create POD redis-master from JSON
  # Pre-condition: no POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f examples/limitrange/valid-pod.json "${kube_flags[@]}"
  # Post-condition: valid-pod POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Delete POD valid-pod with label
  # Pre-condition: valid-pod POD is running
  kube::test::get_object_assert "pods -l'name in (valid-pod)'" '{{range.items}}{{$id_field}}:{{end}}' 'valid-pod:'
  # Command
  kubectl delete pods -l'name in (valid-pod)' "${kube_flags[@]}"
  # Post-condition: no POD is running
  kube::test::get_object_assert "pods -l'name in (valid-pod)'" '{{range.items}}{{$id_field}}:{{end}}' ''

  ### Create POD valid-pod from JSON
  # Pre-condition: no POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f examples/limitrange/valid-pod.json "${kube_flags[@]}"
  # Post-condition: valid-pod POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Delete PODs with no parameter mustn't kill everything
  # Pre-condition: valid-pod POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  ! kubectl delete pods "${kube_flags[@]}"
  # Post-condition: valid-pod POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Delete PODs with --all and a label selector is not permitted
  # Pre-condition: valid-pod POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  ! kubectl delete --all pods -l'name in (valid-pod)' "${kube_flags[@]}"
  # Post-condition: valid-pod POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Delete all PODs
  # Pre-condition: valid-pod POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete --all pods "${kube_flags[@]}" # --all remove all the pods
  # Post-condition: no POD is running
  kube::test::get_object_assert "pods -l'name in (valid-pod)'" '{{range.items}}{{$id_field}}:{{end}}' ''

  ### Create two PODs
  # Pre-condition: no POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f examples/limitrange/valid-pod.json "${kube_flags[@]}"
  kubectl create -f examples/redis/redis-proxy.yaml "${kube_flags[@]}"
  # Post-condition: valid-pod and redis-proxy PODs are running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'redis-proxy:valid-pod:'

  ### Delete multiple PODs at once
  # Pre-condition: valid-pod and redis-proxy PODs are running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'redis-proxy:valid-pod:'
  # Command
  kubectl delete pods valid-pod redis-proxy "${kube_flags[@]}" # delete multiple pods at once
  # Post-condition: no POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create two PODs
  # Pre-condition: no POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f examples/limitrange/valid-pod.json "${kube_flags[@]}"
  kubectl create -f examples/redis/redis-proxy.yaml "${kube_flags[@]}"
  # Post-condition: valid-pod and redis-proxy PODs are running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'redis-proxy:valid-pod:'

  ### Stop multiple PODs at once
  # Pre-condition: valid-pod and redis-proxy PODs are running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'redis-proxy:valid-pod:'
  # Command
  kubectl stop pods valid-pod redis-proxy "${kube_flags[@]}" # stop multiple pods at once
  # Post-condition: no POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create valid-pod POD
  # Pre-condition: no POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f examples/limitrange/valid-pod.json "${kube_flags[@]}"
  # Post-condition: valid-pod POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Label the valid-pod POD
  # Pre-condition: valid-pod is not labelled
  kube::test::get_object_assert 'pod valid-pod' "{{range$labels_field}}{{.}}:{{end}}" 'valid-pod:'
  # Command
  kubectl label pods valid-pod new-name=new-valid-pod "${kube_flags[@]}"
  # Post-conditon: valid-pod is labelled
  kube::test::get_object_assert 'pod valid-pod' "{{range$labels_field}}{{.}}:{{end}}" 'valid-pod:new-valid-pod:'

  ### Delete POD by label
  # Pre-condition: valid-pod POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete pods -lnew-name=new-valid-pod "${kube_flags[@]}"
  # Post-condition: no POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create valid-pod POD
  # Pre-condition: no POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f examples/limitrange/valid-pod.json "${kube_flags[@]}"
  # Post-condition: valid-pod POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Overwriting an existing label is not permitted
  # Pre-condition: name is valid-pod
  kube::test::get_object_assert 'pod valid-pod' "{{${labels_field}.name}}" 'valid-pod'
  # Command
  ! kubectl label pods valid-pod name=valid-pod-super-sayan "${kube_flags[@]}"
  # Post-condition: name is still valid-pod
  kube::test::get_object_assert 'pod valid-pod' "{{${labels_field}.name}}" 'valid-pod'

  ### --overwrite must be used to overwrite existing label, can be applied to all resources
  # Pre-condition: name is valid-pod
  kube::test::get_object_assert 'pod valid-pod' "{{${labels_field}.name}}" 'valid-pod'
  # Command
  kubectl label --overwrite pods --all name=valid-pod-super-sayan "${kube_flags[@]}"
  # Post-condition: name is valid-pod-super-sayan
  kube::test::get_object_assert 'pod valid-pod' "{{${labels_field}.name}}" 'valid-pod-super-sayan'

  ### Delete POD by label
  # Pre-condition: valid-pod POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete pods -l'name in (valid-pod-super-sayan)' "${kube_flags[@]}"
  # Post-condition: no POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''


  ##############
  # Namespaces #
  ##############

  ### Create POD valid-pod in specific namespace
  # Pre-condition: no POD is running
  kube::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create "${kube_flags[@]}" --namespace=other -f examples/limitrange/valid-pod.json
  # Post-condition: valid-pod POD is running
  kube::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Delete POD valid-pod in specific namespace
  # Pre-condition: valid-pod POD is running
  kube::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete "${kube_flags[@]}" pod --namespace=other valid-pod
  # Post-condition: no POD is running
  kube::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{$id_field}}:{{end}}" ''


  #################
  # Pod templates #
  #################

  # Note: pod templates exist only in v1beta3 and above, so output will always be in that form

  ### Create PODTEMPLATE
  # Pre-condition: no PODTEMPLATE
  kube::test::get_object_assert podtemplates "{{range.items}}{{.metadata.name}}:{{end}}" ''
  # Command
  kubectl create -f examples/walkthrough/podtemplate.json "${kube_flags[@]}"
  # Post-condition: nginx PODTEMPLATE is available
  kube::test::get_object_assert podtemplates "{{range.items}}{{.metadata.name}}:{{end}}" 'nginx:'

  ### Printing pod templates works
  kubectl get podtemplates "${kube_flags[@]}"
  ### Display of an object which doesn't existing in v1beta1 and v1beta2 works
  [[ "$(kubectl get podtemplates -o yaml "${kube_flags[@]}" | grep nginx)" ]]

  ### Delete nginx pod template by name
  # Pre-condition: nginx pod template is available
  kube::test::get_object_assert podtemplates "{{range.items}}{{.metadata.name}}:{{end}}" 'nginx:'
  # Command
  kubectl delete podtemplate nginx "${kube_flags[@]}"
  # Post-condition: No templates exist
  kube::test::get_object_assert podtemplate "{{range.items}}{{.metadata.name}}:{{end}}" ''


  ############
  # Services #
  ############

  kube::log::status "Testing kubectl(${version}:services)"

  ### Create redis-master service from JSON
  # Pre-condition: Only the default kubernetes services are running
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:kubernetes-ro:'
  # Command
  kubectl create -f examples/guestbook/redis-master-service.json "${kube_flags[@]}"
  # Post-condition: redis-master service is running
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:kubernetes-ro:redis-master:'
  # Describe command should print detailed information
  kube::test::describe_object_assert services 'redis-master' "Name:" "Labels:" "Selector:" "IP:" "Port:" "Endpoints:" "Session Affinity:"

  ### Dump current redis-master service
  output_service=$(kubectl get service redis-master -o json --output-version=v1beta3 "${kube_flags[@]}")

  ### Delete redis-master-service by id
  # Pre-condition: redis-master service is running
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:kubernetes-ro:redis-master:'
  # Command
  kubectl delete service redis-master "${kube_flags[@]}"
  # Post-condition: Only the default kubernetes services are running
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:kubernetes-ro:'

  ### Create redis-master-service from dumped JSON
  # Pre-condition: Only the default kubernetes services are running
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:kubernetes-ro:'
  # Command
  echo "${output_service}" | kubectl create -f - "${kube_flags[@]}"
  # Post-condition: redis-master service is running
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:kubernetes-ro:redis-master:'

  ### Create redis-master-${version}-test service
  # Pre-condition: redis-master-service service is running
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:kubernetes-ro:redis-master:'
  # Command
  kubectl create -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "Service",
  "apiVersion": "v1beta3",
  "metadata": {
    "name": "service-${version}-test"
  },
  "spec": {
    "ports": [
      {
        "protocol": "TCP",
        "port": 80,
        "targetPort": 80
      }
    ]
  }
}
__EOF__
  # Post-condition:redis-master-service service is running
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:kubernetes-ro:redis-master:service-.*-test:'

  # Command
  kubectl update service "${kube_flags[@]}" service-${version}-test --patch="{\"spec\":{\"selector\":{\"my\":\"test-label\"}},\"apiVersion\":\"v1beta3\"}" --api-version=v1beta3
  # Post-condition: selector has "test-label" label.
  kube::test::get_object_assert "service service-${version}-test" "{{range$service_selector_field}}{{.}}{{end}}" "test-label"

  ### Identity
  kubectl get service "${kube_flags[@]}" service-${version}-test -o json | kubectl update "${kube_flags[@]}" -f -

  ### Delete services by id
  # Pre-condition: redis-master-service service is running
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:kubernetes-ro:redis-master:service-.*-test:'
  # Command
  kubectl delete service redis-master "${kube_flags[@]}"
  kubectl delete service "service-${version}-test" "${kube_flags[@]}"
  # Post-condition: Only the default kubernetes services are running
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:kubernetes-ro:'

  ### Create two services
  # Pre-condition: Only the default kubernetes services are running
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:kubernetes-ro:'
  # Command
  kubectl create -f examples/guestbook/redis-master-service.json "${kube_flags[@]}"
  kubectl create -f examples/guestbook/redis-slave-service.json "${kube_flags[@]}"
  # Post-condition: redis-master and redis-slave services are running
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:kubernetes-ro:redis-master:redis-slave:'

  ### Delete multiple services at once
  # Pre-condition: redis-master and redis-slave services are running
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:kubernetes-ro:redis-master:redis-slave:'
  # Command
  kubectl delete services redis-master redis-slave "${kube_flags[@]}" # delete multiple services at once
  # Post-condition: Only the default kubernetes services are running
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:kubernetes-ro:'


  ###########################
  # Replication controllers #
  ###########################

  kube::log::status "Testing kubectl(${version}:replicationcontrollers)"

  ### Create replication controller frontend from JSON
  # Pre-condition: no replication controller is running
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f examples/guestbook/frontend-controller.json "${kube_flags[@]}"
  # Post-condition: frontend replication controller is running
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'frontend:'
  # Describe command should print detailed information
  kube::test::describe_object_assert rc 'frontend' "Name:" "Image(s):" "Labels:" "Selector:" "Replicas:" "Pods Status:"

  ### Resize replication controller frontend with current-replicas and replicas
  # Pre-condition: 3 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '3'
  # Command
  kubectl resize --current-replicas=3 --replicas=2 replicationcontrollers frontend "${kube_flags[@]}"
  # Post-condition: 2 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '2'

  ### Resize replication controller frontend with (wrong) current-replicas and replicas
  # Pre-condition: 2 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '2'
  # Command
  ! kubectl resize --current-replicas=3 --replicas=2 replicationcontrollers frontend "${kube_flags[@]}"
  # Post-condition: nothing changed
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '2'

  ### Resize replication controller frontend with replicas only
  # Pre-condition: 2 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '2'
  # Command
  kubectl resize  --replicas=3 replicationcontrollers frontend "${kube_flags[@]}"
  # Post-condition: 3 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '3'

  ### Expose replication controller as service
  # Pre-condition: 3 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '3'
  # Command
  kubectl expose rc frontend --port=80 "${kube_flags[@]}"
  # Post-condition: service exists
  kube::test::get_object_assert 'service frontend' "{{$port_field}}" '80'
  # Command
  kubectl expose service frontend --port=443 --service-name=frontend-2 "${kube_flags[@]}"
  # Post-condition: service exists
  kube::test::get_object_assert 'service frontend-2' "{{$port_field}}" '443'
  # Command
  kubectl create -f examples/limitrange/valid-pod.json "${kube_flags[@]}"
  kubectl expose pod valid-pod --port=444 --service-name=frontend-3 "${kube_flags[@]}"
  # Post-condition: service exists
  kube::test::get_object_assert 'service frontend-3' "{{$port_field}}" '444'
  # Cleanup services
  kubectl delete pod valid-pod "${kube_flags[@]}"
  kubectl delete service frontend{,-2,-3} "${kube_flags[@]}"

  ### Perform a rolling update with --image
  # Command
  kubectl rolling-update frontend --image=kubernetes/pause --update-period=10ns --poll-interval=10ms "${kube_flags[@]}"
  # Post-condition: current image IS kubernetes/pause
  kube::test::get_object_assert 'rc frontend' '{{range \$c:=$rc_container_image_field}} {{\$c.image}} {{end}}' ' +kubernetes/pause +'

  ### Delete replication controller with id
  # Pre-condition: frontend replication controller is running
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'frontend:'
  # Command
  kubectl stop rc frontend "${kube_flags[@]}"
  # Post-condition: no replication controller is running
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create two replication controllers
  # Pre-condition: no replication controller is running
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f examples/guestbook/frontend-controller.json "${kube_flags[@]}"
  kubectl create -f examples/guestbook/redis-slave-controller.json "${kube_flags[@]}"
  # Post-condition: frontend and redis-slave
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'frontend:redis-slave:'

  ### Delete multiple controllers at once
  # Pre-condition: frontend and redis-slave
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'frontend:redis-slave:'
  # Command
  kubectl stop rc frontend redis-slave "${kube_flags[@]}" # delete multiple controllers at once
  # Post-condition: no replication controller is running
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''

  ######################
  # Persistent Volumes #
  ######################

  ### Create and delete persistent volume examples
  # Pre-condition: no persistent volumes currently exist
  kube::test::get_object_assert pv "{{range.items}}{{.$id_field}}:{{end}}" ''
  # Command
  kubectl create -f examples/persistent-volumes/volumes/local-01.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pv "{{range.items}}{{.$id_field}}:{{end}}" 'pv0001:'
  kubectl delete pv pv0001 "${kube_flags[@]}"
  kubectl create -f examples/persistent-volumes/volumes/local-02.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pv "{{range.items}}{{.$id_field}}:{{end}}" 'pv0002:'
  kubectl delete pv pv0002 "${kube_flags[@]}"
  kubectl create -f examples/persistent-volumes/volumes/gce.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pv "{{range.items}}{{.$id_field}}:{{end}}" 'pv0003:'
  kubectl delete pv pv0003 "${kube_flags[@]}"
  # Post-condition: no PVs
  kube::test::get_object_assert pv "{{range.items}}{{.$id_field}}:{{end}}" ''

  ############################
  # Persistent Volume Claims #
  ############################

  ### Create and delete persistent volume claim examples
  # Pre-condition: no persistent volume claims currently exist
  kube::test::get_object_assert pvc "{{range.items}}{{.$id_field}}:{{end}}" ''
  # Command
  kubectl create -f examples/persistent-volumes/claims/claim-01.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pvc "{{range.items}}{{.$id_field}}:{{end}}" 'myclaim-1:'
  kubectl delete pvc myclaim-1 "${kube_flags[@]}"

  kubectl create -f examples/persistent-volumes/claims/claim-02.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pvc "{{range.items}}{{.$id_field}}:{{end}}" 'myclaim-2:'
  kubectl delete pvc myclaim-2 "${kube_flags[@]}"

  kubectl create -f examples/persistent-volumes/claims/claim-03.json "${kube_flags[@]}"
  kube::test::get_object_assert pvc "{{range.items}}{{.$id_field}}:{{end}}" 'myclaim-3:'
  kubectl delete pvc myclaim-3 "${kube_flags[@]}"
  # Post-condition: no PVCs
  kube::test::get_object_assert pvc "{{range.items}}{{.$id_field}}:{{end}}" ''



  #########
  # Nodes #
  #########

  kube::log::status "Testing kubectl(${version}:nodes)"

  kube::test::get_object_assert nodes "{{range.items}}{{$id_field}}:{{end}}" '127.0.0.1:'

  kube::test::describe_object_assert nodes "127.0.0.1" "Name:" "Labels:" "CreationTimestamp:" "Conditions:" "Addresses:" "Capacity:" "Pods:"


  ###########
  # Minions #
  ###########

  if [[ "${version}" = "v1beta1" ]] || [[ "${version}" = "v1beta2" ]]; then
    kube::log::status "Testing kubectl(${version}:minions)"

    kube::test::get_object_assert minions "{{range.items}}{{$id_field}}:{{end}}" '127.0.0.1:'

    # TODO: I should be a MinionList instead of List
    kube::test::get_object_assert minions '{{.kind}}' 'List'

    kube::test::describe_object_assert minions "127.0.0.1" "Name:" "Conditions:" "Addresses:" "Capacity:" "Pods:"
  fi


  #####################
  # Retrieve multiple #
  #####################

  kube::log::status "Testing kubectl(${version}:multiget)"
  kube::test::get_object_assert 'nodes/127.0.0.1 service/kubernetes' "{{range.items}}{{$id_field}}:{{end}}" '127.0.0.1:kubernetes:'


  #####################
  # Resource aliasing #
  #####################

  kube::log::status "Testing resource aliasing"
  kubectl create -f examples/cassandra/cassandra.yaml "${kube_flags[@]}"
  kubectl create -f examples/cassandra/cassandra-controller.yaml "${kube_flags[@]}"
  kubectl create -f examples/cassandra/cassandra-service.yaml "${kube_flags[@]}"
  kube::test::get_object_assert "all -l'name=cassandra'" "{{range.items}}{{$id_field}}:{{end}}" 'cassandra:cassandra:cassandra:'
  kubectl delete all -l name=cassandra "${kube_flags[@]}"


  ###########
  # Swagger #
  ###########

  if [[ -n "${version}" ]]; then
    # Verify schema
    file="${KUBE_TEMP}/schema-${version}.json"
    curl -s "http://127.0.0.1:${API_PORT}/swaggerapi/api/${version}" > "${file}"
    [[ "$(grep "list of returned" "${file}")" ]]
    [[ "$(grep "list of pods" "${file}")" ]]
    [[ "$(grep "watch for changes to the described resources" "${file}")" ]]
  fi

  kube::test::clear_all
done

kube::log::status "TEST PASSED"
