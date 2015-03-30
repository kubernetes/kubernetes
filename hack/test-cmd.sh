#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

LMKTFY_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${LMKTFY_ROOT}/hack/lib/init.sh"
source "${LMKTFY_ROOT}/hack/lib/test.sh"

function cleanup()
{
    [[ -n ${APISERVER_PID-} ]] && kill ${APISERVER_PID} 1>&2 2>/dev/null
    [[ -n ${CTLRMGR_PID-} ]] && kill ${CTLRMGR_PID} 1>&2 2>/dev/null
    [[ -n ${LMKTFYLET_PID-} ]] && kill ${LMKTFYLET_PID} 1>&2 2>/dev/null
    [[ -n ${PROXY_PID-} ]] && kill ${PROXY_PID} 1>&2 2>/dev/null

    lmktfy::etcd::cleanup
    rm -rf "${LMKTFY_TEMP}"

    lmktfy::log::status "Clean up complete"
}

trap cleanup EXIT SIGINT

lmktfy::util::ensure-temp-dir
lmktfy::etcd::start

ETCD_HOST=${ETCD_HOST:-127.0.0.1}
ETCD_PORT=${ETCD_PORT:-4001}
API_PORT=${API_PORT:-8080}
API_HOST=${API_HOST:-127.0.0.1}
LMKTFYLET_PORT=${LMKTFYLET_PORT:-10250}
CTLRMGR_PORT=${CTLRMGR_PORT:-10252}

# Check lmktfyctl
lmktfy::log::status "Running lmktfyctl with no options"
"${LMKTFY_OUTPUT_HOSTBIN}/lmktfyctl"

lmktfy::log::status "Starting lmktfylet in masterless mode"
"${LMKTFY_OUTPUT_HOSTBIN}/lmktfylet" \
  --really_crash_for_testing=true \
  --root_dir=/tmp/lmktfylet.$$ \
  --docker_endpoint="fake://" \
  --hostname_override="127.0.0.1" \
  --address="127.0.0.1" \
  --port="$LMKTFYLET_PORT" 1>&2 &
LMKTFYLET_PID=$!
lmktfy::util::wait_for_url "http://127.0.0.1:${LMKTFYLET_PORT}/healthz" "lmktfylet: "
kill ${LMKTFYLET_PID} 1>&2 2>/dev/null

lmktfy::log::status "Starting lmktfylet in masterful mode"
"${LMKTFY_OUTPUT_HOSTBIN}/lmktfylet" \
  --really_crash_for_testing=true \
  --root_dir=/tmp/lmktfylet.$$ \
  --docker_endpoint="fake://" \
  --hostname_override="127.0.0.1" \
  --address="127.0.0.1" \
  --api_servers="${API_HOST}:${API_PORT}" \
  --auth_path="${LMKTFY_ROOT}/hack/.test-cmd-auth" \
  --port="$LMKTFYLET_PORT" 1>&2 &
LMKTFYLET_PID=$!

lmktfy::util::wait_for_url "http://127.0.0.1:${LMKTFYLET_PORT}/healthz" "lmktfylet: "

# Start lmktfy-apiserver
lmktfy::log::status "Starting lmktfy-apiserver"
"${LMKTFY_OUTPUT_HOSTBIN}/lmktfy-apiserver" \
  --address="127.0.0.1" \
  --public_address_override="127.0.0.1" \
  --port="${API_PORT}" \
  --etcd_servers="http://${ETCD_HOST}:${ETCD_PORT}" \
  --public_address_override="127.0.0.1" \
  --lmktfylet_port=${LMKTFYLET_PORT} \
  --runtime_config=api/v1beta3 \
  --portal_net="10.0.0.0/24" 1>&2 &
APISERVER_PID=$!

lmktfy::util::wait_for_url "http://127.0.0.1:${API_PORT}/healthz" "apiserver: "

# Start controller manager
lmktfy::log::status "Starting CONTROLLER-MANAGER"
"${LMKTFY_OUTPUT_HOSTBIN}/lmktfy-controller-manager" \
  --machines="127.0.0.1" \
  --master="127.0.0.1:${API_PORT}" 1>&2 &
CTLRMGR_PID=$!

lmktfy::util::wait_for_url "http://127.0.0.1:${CTLRMGR_PORT}/healthz" "controller-manager: "
lmktfy::util::wait_for_url "http://127.0.0.1:${API_PORT}/api/v1beta1/minions/127.0.0.1" "apiserver(minions): " 0.2 25

# Expose lmktfyctl directly for readability
PATH="${LMKTFY_OUTPUT_HOSTBIN}":$PATH

lmktfy_api_versions=(
  ""
  v1beta1
  v1beta2
  v1beta3
)
for version in "${lmktfy_api_versions[@]}"; do
  if [[ -z "${version}" ]]; then
    lmktfy_flags=(
      -s "http://127.0.0.1:${API_PORT}"
      --match-server-version
    )
    [ "$(lmktfyctl get minions -t $'{{ .apiVersion }}' "${lmktfy_flags[@]}")" == "v1beta1" ]
  else
    lmktfy_flags=(
      -s "http://127.0.0.1:${API_PORT}"
      --match-server-version
      --api-version="${version}"
    )
    [ "$(lmktfyctl get minions -t $'{{ .apiVersion }}' "${lmktfy_flags[@]}")" == "${version}" ]
  fi
  id_field="id"
  labels_field="labels"
  service_selector_field="selector"
  rc_replicas_field="desiredState.replicas"
  port_field="port"
  if [ "$version" = "v1beta3" ]; then
    id_field="metadata.name"
    labels_field="metadata.labels"
    service_selector_field="spec.selector"
    rc_replicas_field="spec.replicas"
    port_field="spec.port"
  fi

  # Passing no arguments to create is an error
  ! lmktfyctl create

  ###########################
  # POD creation / deletion #
  ###########################

  lmktfy::log::status "Testing lmktfyctl(${version}:pods)"

  ### Create POD valid-pod from JSON
  # Pre-condition: no POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" ''
  # Command
  lmktfyctl create "${lmktfy_flags[@]}" -f examples/limitrange/valid-pod.json
  # Post-condition: valid-pod POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" 'valid-pod:'
  lmktfy::test::get_object_assert 'pod valid-pod' "{{.$id_field}}" 'valid-pod'
  lmktfy::test::get_object_assert 'pod/valid-pod' "{{.$id_field}}" 'valid-pod'
  lmktfy::test::get_object_assert 'pods/valid-pod' "{{.$id_field}}" 'valid-pod'
  # Describe command should print detailed information
  lmktfy::test::describe_object_assert pods 'valid-pod' "Name:" "Image(s):" "Host:" "Labels:" "Status:" "Replication Controllers"

  ### Dump current valid-pod POD
  output_pod=$(lmktfyctl get pod valid-pod -o yaml --output-version=v1beta1 "${lmktfy_flags[@]}")

  ### Delete POD valid-pod by id
  # Pre-condition: valid-pod POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" 'valid-pod:'
  # Command
  lmktfyctl delete pod valid-pod "${lmktfy_flags[@]}"
  # Post-condition: no POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" ''

  ### Create POD valid-pod from dumped YAML
  # Pre-condition: no POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" ''
  # Command
  echo "${output_pod}" | lmktfyctl create -f - "${lmktfy_flags[@]}"
  # Post-condition: valid-pod POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" 'valid-pod:'

  ### Delete POD valid-pod from JSON
  # Pre-condition: valid-pod POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" 'valid-pod:'
  # Command
  lmktfyctl delete -f examples/limitrange/valid-pod.json "${lmktfy_flags[@]}"
  # Post-condition: no POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" ''

  ### Create POD redis-master from JSON
  # Pre-condition: no POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" ''
  # Command
  lmktfyctl create -f examples/limitrange/valid-pod.json "${lmktfy_flags[@]}"
  # Post-condition: valid-pod POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" 'valid-pod:'

  ### Delete POD valid-pod with label
  # Pre-condition: valid-pod POD is running
  lmktfy::test::get_object_assert "pods -l'name in (valid-pod)'" '{{range.items}}{{.$id_field}}:{{end}}' 'valid-pod:'
  # Command
  lmktfyctl delete pods -l'name in (valid-pod)' "${lmktfy_flags[@]}"
  # Post-condition: no POD is running
  lmktfy::test::get_object_assert "pods -l'name in (valid-pod)'" '{{range.items}}{{.$id_field}}:{{end}}' ''

  ### Create POD valid-pod from JSON
  # Pre-condition: no POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" ''
  # Command
  lmktfyctl create -f examples/limitrange/valid-pod.json "${lmktfy_flags[@]}"
  # Post-condition: valid-pod POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" 'valid-pod:'

  ### Delete PODs with no parameter mustn't kill everything
  # Pre-condition: valid-pod POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" 'valid-pod:'
  # Command
  ! lmktfyctl delete pods "${lmktfy_flags[@]}"
  # Post-condition: valid-pod POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" 'valid-pod:'

  ### Delete PODs with --all and a label selector is not permitted
  # Pre-condition: valid-pod POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" 'valid-pod:'
  # Command
  ! lmktfyctl delete --all pods -l'name in (valid-pod)' "${lmktfy_flags[@]}"
  # Post-condition: valid-pod POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" 'valid-pod:'

  ### Delete all PODs
  # Pre-condition: valid-pod POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" 'valid-pod:'
  # Command
  lmktfyctl delete --all pods "${lmktfy_flags[@]}" # --all remove all the pods
  # Post-condition: no POD is running
  lmktfy::test::get_object_assert "pods -l'name in (valid-pod)'" '{{range.items}}{{.$id_field}}:{{end}}' ''

  ### Create two PODs
  # Pre-condition: no POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" ''
  # Command
  lmktfyctl create -f examples/limitrange/valid-pod.json "${lmktfy_flags[@]}"
  lmktfyctl create -f examples/redis/redis-proxy.yaml "${lmktfy_flags[@]}"
  # Post-condition: valid-pod and redis-proxy PODs are running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" 'redis-proxy:valid-pod:'

  ### Delete multiple PODs at once
  # Pre-condition: valid-pod and redis-proxy PODs are running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" 'redis-proxy:valid-pod:'
  # Command
  lmktfyctl delete pods valid-pod redis-proxy "${lmktfy_flags[@]}" # delete multiple pods at once
  # Post-condition: no POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" ''

  ### Create two PODs
  # Pre-condition: no POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" ''
  # Command
  lmktfyctl create -f examples/limitrange/valid-pod.json "${lmktfy_flags[@]}"
  lmktfyctl create -f examples/redis/redis-proxy.yaml "${lmktfy_flags[@]}"
  # Post-condition: valid-pod and redis-proxy PODs are running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" 'redis-proxy:valid-pod:'

  ### Stop multiple PODs at once
  # Pre-condition: valid-pod and redis-proxy PODs are running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" 'redis-proxy:valid-pod:'
  # Command
  lmktfyctl stop pods valid-pod redis-proxy "${lmktfy_flags[@]}" # stop multiple pods at once
  # Post-condition: no POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" ''

  ### Create valid-pod POD
  # Pre-condition: no POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" ''
  # Command
  lmktfyctl create -f examples/limitrange/valid-pod.json "${lmktfy_flags[@]}"
  # Post-condition: valid-pod POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" 'valid-pod:'

  ### Label the valid-pod POD
  # Pre-condition: valid-pod is not labelled
  lmktfy::test::get_object_assert 'pod valid-pod' "{{range.$labels_field}}{{.}}:{{end}}" 'valid-pod:'
  # Command
  lmktfyctl label pods valid-pod new-name=new-valid-pod "${lmktfy_flags[@]}"
  # Post-conditon: valid-pod is labelled
  lmktfy::test::get_object_assert 'pod valid-pod' "{{range.$labels_field}}{{.}}:{{end}}" 'valid-pod:new-valid-pod:'

  ### Delete POD by label
  # Pre-condition: valid-pod POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" 'valid-pod:'
  # Command
  lmktfyctl delete pods -lnew-name=new-valid-pod "${lmktfy_flags[@]}"
  # Post-condition: no POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" ''

  ### Create valid-pod POD
  # Pre-condition: no POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" ''
  # Command
  lmktfyctl create -f examples/limitrange/valid-pod.json "${lmktfy_flags[@]}"
  # Post-condition: valid-pod POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" 'valid-pod:'

  ### Overwriting an existing label is not permitted
  # Pre-condition: name is valid-pod
  lmktfy::test::get_object_assert 'pod valid-pod' "{{.${labels_field}.name}}" 'valid-pod'
  # Command
  ! lmktfyctl label pods valid-pod name=valid-pod-super-sayan "${lmktfy_flags[@]}"
  # Post-condition: name is still valid-pod
  lmktfy::test::get_object_assert 'pod valid-pod' "{{.${labels_field}.name}}" 'valid-pod'

  ### --overwrite must be used to overwrite existing label, can be applied to all resources
  # Pre-condition: name is valid-pod
  lmktfy::test::get_object_assert 'pod valid-pod' "{{.${labels_field}.name}}" 'valid-pod'
  # Command
  lmktfyctl label --overwrite pods --all name=valid-pod-super-sayan "${lmktfy_flags[@]}"
  # Post-condition: name is valid-pod-super-sayan
  lmktfy::test::get_object_assert 'pod valid-pod' "{{.${labels_field}.name}}" 'valid-pod-super-sayan'

  ### Delete POD by label
  # Pre-condition: valid-pod POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" 'valid-pod:'
  # Command
  lmktfyctl delete pods -l'name in (valid-pod-super-sayan)' "${lmktfy_flags[@]}"
  # Post-condition: no POD is running
  lmktfy::test::get_object_assert pods "{{range.items}}{{.$id_field}}:{{end}}" ''


  ##############
  # Namespaces #
  ##############

  ### Create POD valid-pod in specific namespace
  # Pre-condition: no POD is running
  lmktfy::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{.$id_field}}:{{end}}" ''
  # Command
  lmktfyctl create "${lmktfy_flags[@]}" --namespace=other -f examples/limitrange/valid-pod.json
  # Post-condition: valid-pod POD is running
  lmktfy::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{.$id_field}}:{{end}}" 'valid-pod:'

  ### Delete POD valid-pod in specific namespace
  # Pre-condition: valid-pod POD is running
  lmktfy::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{.$id_field}}:{{end}}" 'valid-pod:'
  # Command
  lmktfyctl delete "${lmktfy_flags[@]}" pod --namespace=other valid-pod
  # Post-condition: no POD is running
  lmktfy::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{.$id_field}}:{{end}}" ''


  ############
  # Services #
  ############

  lmktfy::log::status "Testing lmktfyctl(${version}:services)"

  ### Create redis-master service from JSON
  # Pre-condition: Only the default lmktfy services are running
  lmktfy::test::get_object_assert services "{{range.items}}{{.$id_field}}:{{end}}" 'lmktfy:lmktfy-ro:'
  # Command
  lmktfyctl create -f examples/guestbook/redis-master-service.json "${lmktfy_flags[@]}"
  # Post-condition: redis-master service is running
  lmktfy::test::get_object_assert services "{{range.items}}{{.$id_field}}:{{end}}" 'lmktfy:lmktfy-ro:redis-master:'
  # Describe command should print detailed information
  lmktfy::test::describe_object_assert services 'redis-master' "Name:" "Labels:" "Selector:" "IP:" "Port:" "Endpoints:" "Session Affinity:"

  ### Dump current redis-master service
  output_service=$(lmktfyctl get service redis-master -o json --output-version=v1beta3 "${lmktfy_flags[@]}")

  ### Delete redis-master-service by id
  # Pre-condition: redis-master service is running
  lmktfy::test::get_object_assert services "{{range.items}}{{.$id_field}}:{{end}}" 'lmktfy:lmktfy-ro:redis-master:'
  # Command
  lmktfyctl delete service redis-master "${lmktfy_flags[@]}"
  # Post-condition: Only the default lmktfy services are running
  lmktfy::test::get_object_assert services "{{range.items}}{{.$id_field}}:{{end}}" 'lmktfy:lmktfy-ro:'

  ### Create redis-master-service from dumped JSON
  # Pre-condition: Only the default lmktfy services are running
  lmktfy::test::get_object_assert services "{{range.items}}{{.$id_field}}:{{end}}" 'lmktfy:lmktfy-ro:'
  # Command
  echo "${output_service}" | lmktfyctl create -f - "${lmktfy_flags[@]}"
  # Post-condition: redis-master service is running
  lmktfy::test::get_object_assert services "{{range.items}}{{.$id_field}}:{{end}}" 'lmktfy:lmktfy-ro:redis-master:'

  ### Create redis-master-${version}-test service
  # Pre-condition: redis-master-service service is running
  lmktfy::test::get_object_assert services "{{range.items}}{{.$id_field}}:{{end}}" 'lmktfy:lmktfy-ro:redis-master:'
  # Command
  lmktfyctl create -f - "${lmktfy_flags[@]}" << __EOF__
      {
          "kind": "Service",
          "apiVersion": "v1beta1",
          "id": "service-${version}-test",
          "port": 80,
          "protocol": "TCP"
      }
__EOF__
  # Post-condition:redis-master-service service is running
  lmktfy::test::get_object_assert services "{{range.items}}{{.$id_field}}:{{end}}" 'lmktfy:lmktfy-ro:redis-master:service-.*-test:'

  # Command
  lmktfyctl update service "${lmktfy_flags[@]}" service-${version}-test --patch="{\"selector\":{\"my\":\"test-label\"},\"apiVersion\":\"v1beta1\"}"
  # Post-condition: selector.version == ${version}
  # This test works only in v1beta1 and v1beta2
  # https://github.com/GoogleCloudPlatform/lmktfy/issues/4771
  lmktfy::test::get_object_assert "service service-${version}-test" "{{range.$service_selector_field}}{{.}}{{end}}" "test-label"

  ### Identity
  lmktfyctl get service "${lmktfy_flags[@]}" service-${version}-test -o json | lmktfyctl update "${lmktfy_flags[@]}" -f -

  ### Delete services by id
  # Pre-condition: redis-master-service service is running
  lmktfy::test::get_object_assert services "{{range.items}}{{.$id_field}}:{{end}}" 'lmktfy:lmktfy-ro:redis-master:service-.*-test:'
  # Command
  lmktfyctl delete service redis-master "${lmktfy_flags[@]}"
  lmktfyctl delete service "service-${version}-test" "${lmktfy_flags[@]}"
  # Post-condition: Only the default lmktfy services are running
  lmktfy::test::get_object_assert services "{{range.items}}{{.$id_field}}:{{end}}" 'lmktfy:lmktfy-ro:'

  ### Create two services
  # Pre-condition: Only the default lmktfy services are running
  lmktfy::test::get_object_assert services "{{range.items}}{{.$id_field}}:{{end}}" 'lmktfy:lmktfy-ro:'
  # Command
  lmktfyctl create -f examples/guestbook/redis-master-service.json "${lmktfy_flags[@]}"
  lmktfyctl create -f examples/guestbook/redis-slave-service.json "${lmktfy_flags[@]}"
  # Post-condition: redis-master and redis-slave services are running
  lmktfy::test::get_object_assert services "{{range.items}}{{.$id_field}}:{{end}}" 'lmktfy:lmktfy-ro:redis-master:redis-slave:'

  ### Delete multiple services at once
  # Pre-condition: redis-master and redis-slave services are running
  lmktfy::test::get_object_assert services "{{range.items}}{{.$id_field}}:{{end}}" 'lmktfy:lmktfy-ro:redis-master:redis-slave:'
  # Command
  lmktfyctl delete services redis-master redis-slave "${lmktfy_flags[@]}" # delete multiple services at once
  # Post-condition: Only the default lmktfy services are running
  lmktfy::test::get_object_assert services "{{range.items}}{{.$id_field}}:{{end}}" 'lmktfy:lmktfy-ro:'


  ###########################
  # Replication controllers #
  ###########################

  lmktfy::log::status "Testing lmktfyctl(${version}:replicationcontrollers)"

  ### Create replication controller frontend from JSON
  # Pre-condition: no replication controller is running
  lmktfy::test::get_object_assert rc "{{range.items}}{{.$id_field}}:{{end}}" ''
  # Command
  lmktfyctl create -f examples/guestbook/frontend-controller.json "${lmktfy_flags[@]}"
  # Post-condition: frontend replication controller is running
  lmktfy::test::get_object_assert rc "{{range.items}}{{.$id_field}}:{{end}}" 'frontend-controller:'
  # Describe command should print detailed information
  lmktfy::test::describe_object_assert rc 'frontend-controller' "Name:" "Image(s):" "Labels:" "Selector:" "Replicas:" "Pods Status:"

  ### Resize replication controller frontend with current-replicas and replicas
  # Pre-condition: 3 replicas
  lmktfy::test::get_object_assert 'rc frontend-controller' "{{.$rc_replicas_field}}" '3'
  # Command
  lmktfyctl resize --current-replicas=3 --replicas=2 replicationcontrollers frontend-controller "${lmktfy_flags[@]}"
  # Post-condition: 2 replicas
  lmktfy::test::get_object_assert 'rc frontend-controller' "{{.$rc_replicas_field}}" '2'

  ### Resize replication controller frontend with (wrong) current-replicas and replicas
  # Pre-condition: 2 replicas
  lmktfy::test::get_object_assert 'rc frontend-controller' "{{.$rc_replicas_field}}" '2'
  # Command
  ! lmktfyctl resize --current-replicas=3 --replicas=2 replicationcontrollers frontend-controller "${lmktfy_flags[@]}"
  # Post-condition: nothing changed
  lmktfy::test::get_object_assert 'rc frontend-controller' "{{.$rc_replicas_field}}" '2'

  ### Resize replication controller frontend with replicas only
  # Pre-condition: 2 replicas
  lmktfy::test::get_object_assert 'rc frontend-controller' "{{.$rc_replicas_field}}" '2'
  # Command
  lmktfyctl resize  --replicas=3 replicationcontrollers frontend-controller "${lmktfy_flags[@]}"
  # Post-condition: 3 replicas
  lmktfy::test::get_object_assert 'rc frontend-controller' "{{.$rc_replicas_field}}" '3'

  ### Expose replication controller as service
  # Pre-condition: 3 replicas
  lmktfy::test::get_object_assert 'rc frontend-controller' "{{.$rc_replicas_field}}" '3'
  # Command
  lmktfyctl expose rc frontend-controller --port=80 "${lmktfy_flags[@]}"
  # Post-condition: service exists
  lmktfy::test::get_object_assert 'service frontend-controller' "{{.$port_field}}" '80'
  # Command
  lmktfyctl expose service frontend-controller --port=443 --service-name=frontend-controller-2 "${lmktfy_flags[@]}"
  # Post-condition: service exists
  lmktfy::test::get_object_assert 'service frontend-controller-2' "{{.$port_field}}" '443'
  # Command
  lmktfyctl create -f examples/limitrange/valid-pod.json "${lmktfy_flags[@]}"
  lmktfyctl expose pod valid-pod --port=444 --service-name=frontend-controller-3 "${lmktfy_flags[@]}"
  # Post-condition: service exists
  lmktfy::test::get_object_assert 'service frontend-controller-3' "{{.$port_field}}" '444'
  # Cleanup services
  lmktfyctl delete pod valid-pod "${lmktfy_flags[@]}"
  lmktfyctl delete service frontend-controller{,-2,-3} "${lmktfy_flags[@]}"

  ### Delete replication controller with id
  # Pre-condition: frontend replication controller is running
  lmktfy::test::get_object_assert rc "{{range.items}}{{.$id_field}}:{{end}}" 'frontend-controller:'
  # Command
  lmktfyctl delete rc frontend-controller "${lmktfy_flags[@]}"
  # Post-condition: no replication controller is running
  lmktfy::test::get_object_assert rc "{{range.items}}{{.$id_field}}:{{end}}" ''

  ### Create two replication controllers
  # Pre-condition: no replication controller is running
  lmktfy::test::get_object_assert rc "{{range.items}}{{.$id_field}}:{{end}}" ''
  # Command
  lmktfyctl create -f examples/guestbook/frontend-controller.json "${lmktfy_flags[@]}"
  lmktfyctl create -f examples/guestbook/redis-slave-controller.json "${lmktfy_flags[@]}"
  # Post-condition: frontend and redis-slave
  lmktfy::test::get_object_assert rc "{{range.items}}{{.$id_field}}:{{end}}" 'frontend-controller:redis-slave-controller:'

  ### Delete multiple controllers at once
  # Pre-condition: frontend and redis-slave
  lmktfy::test::get_object_assert rc "{{range.items}}{{.$id_field}}:{{end}}" 'frontend-controller:redis-slave-controller:'
  # Command
  lmktfyctl delete rc frontend-controller redis-slave-controller "${lmktfy_flags[@]}" # delete multiple controllers at once
  # Post-condition: no replication controller is running
  lmktfy::test::get_object_assert rc "{{range.items}}{{.$id_field}}:{{end}}" ''


  #########
  # Nodes #
  #########

  lmktfy::log::status "Testing lmktfyctl(${version}:nodes)"

  lmktfy::test::get_object_assert nodes "{{range.items}}{{.$id_field}}:{{end}}" '127.0.0.1:'

  lmktfy::test::describe_object_assert nodes "127.0.0.1" "Name:" "Labels:" "CreationTimestamp:" "Conditions:" "Addresses:" "Capacity:" "Pods:"


  ###########
  # Minions #
  ###########

  if [[ "${version}" != "v1beta3" ]]; then
    lmktfy::log::status "Testing lmktfyctl(${version}:minions)"

    lmktfy::test::get_object_assert minions "{{range.items}}{{.$id_field}}:{{end}}" '127.0.0.1:'

    # TODO: I should be a MinionList instead of List
    lmktfy::test::get_object_assert minions '{{.kind}}' 'List'

    lmktfy::test::describe_object_assert minions "127.0.0.1" "Name:" "Conditions:" "Addresses:" "Capacity:" "Pods:"
  fi


  #####################
  # Retrieve multiple #
  #####################

  lmktfy::log::status "Testing lmktfyctl(${version}:multiget)"
  lmktfy::test::get_object_assert 'nodes/127.0.0.1 service/lmktfy' "{{range.items}}{{.$id_field}}:{{end}}" '127.0.0.1:lmktfy:'


  ###########
  # Swagger #
  ###########

  if [[ -n "${version}" ]]; then
    # Verify schema
    file="${LMKTFY_TEMP}/schema-${version}.json"
    curl -s "http://127.0.0.1:${API_PORT}/swaggerapi/api/${version}" > "${file}"
    [[ "$(grep "list of returned" "${file}")" ]]
    [[ "$(grep "list of pods" "${file}")" ]]
    [[ "$(grep "watch for changes to the described resources" "${file}")" ]]
  fi

  lmktfy::test::clear_all
done

lmktfy::log::status "TEST PASSED"
