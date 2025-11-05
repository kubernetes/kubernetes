#!/usr/bin/env bash

# Copyright 2025 The Kubernetes Authors.
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

run_kuberc_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl alpha kuberc set commands"

  KUBERC_FILE="${TMPDIR:-/tmp}"/kuberc_file
  cat > "$KUBERC_FILE" << EOF
apiVersion: kubectl.config.k8s.io/v1beta1
kind: Preference
EOF

  # Build up the kuberc file using kubectl alpha kuberc set commands
  kubectl alpha kuberc set --kuberc="$KUBERC_FILE" --section=defaults --command=apply --option=server-side=true --option=dry-run=server --option=validate=strict
  kubectl alpha kuberc set --kuberc="$KUBERC_FILE" --section=defaults --command=delete --option=interactive=true
  kubectl alpha kuberc set --kuberc="$KUBERC_FILE" --section=defaults --command=get --option=namespace=test-kuberc-ns --option=output=json

  kubectl alpha kuberc set --kuberc="$KUBERC_FILE" --section=aliases --name=crns --command="create namespace" --appendarg=test-kuberc-ns
  kubectl alpha kuberc set --kuberc="$KUBERC_FILE" --section=aliases --name=getn --command=get --prependarg=namespace --option=output=wide
  kubectl alpha kuberc set --kuberc="$KUBERC_FILE" --section=aliases --name=crole --command="create role" --option=verb=get,watch
  kubectl alpha kuberc set --kuberc="$KUBERC_FILE" --section=aliases --name=getrole --command=get --option=output=json
  kubectl alpha kuberc set --kuberc="$KUBERC_FILE" --section=aliases --name=runx --command=run --option=image=nginx --option=labels=app=test,env=test --option=env=DNS_DOMAIN=test --option=namespace=test-kuberc-ns --appendarg=test-pod-2 --appendarg=-- --appendarg=custom-arg1 --appendarg=custom-arg2
  kubectl alpha kuberc set --kuberc="$KUBERC_FILE" --section=aliases --name=setx --command="set image" --appendarg=pod/test-pod-2 --appendarg=test-pod-2=busybox

  kube::log::status "Testing kubectl alpha kuberc view commands"
  # Test: kubectl alpha kuberc view
  output_message=$(kubectl alpha kuberc view --kuberc="$KUBERC_FILE")
  kube::test::if_has_string "${output_message}" "apiVersion: kubectl.config.k8s.io/v1beta1"
  kube::test::if_has_string "${output_message}" "kind: Preference"
  kube::test::if_has_string "${output_message}" "command: apply"
  kube::test::if_has_string "${output_message}" "name: runx"
  kube::test::if_has_string "${output_message}" "server-side"
  kube::test::if_has_string "${output_message}" "interactive"

  # Test: kubectl alpha kuberc view with json output
  output_message=$(kubectl alpha kuberc view --kuberc="$KUBERC_FILE" -o json)
  kube::test::if_has_string "${output_message}" "\"apiVersion\": \"kubectl.config.k8s.io/v1beta1\""
  kube::test::if_has_string "${output_message}" "\"kind\": \"Preference\""

  # Test: Attempt to set existing default without --overwrite flag should fail
  output_message=$(! kubectl alpha kuberc set --kuberc="$KUBERC_FILE" --section=defaults --command=get --option=output=yaml 2>&1)
  kube::test::if_has_string "${output_message}" "defaults for command \"get\" already exist, use --overwrite to replace"

  # Test: Now set with --overwrite flag should succeed and merge options
  kubectl alpha kuberc set --kuberc="$KUBERC_FILE" --section=defaults --command=get --option=output=yaml --overwrite
  output_message=$(kubectl alpha kuberc view --kuberc="$KUBERC_FILE")
  kube::test::if_has_string "${output_message}" "default: yaml"
  # Should still have namespace option from before
  kube::test::if_has_string "${output_message}" "default: test-kuberc-ns"

  # Test: Attempt to set existing alias without --overwrite flag should fail
  output_message=$(! kubectl alpha kuberc set --kuberc="$KUBERC_FILE" --section=aliases --name=getn --command=get --prependarg=pods 2>&1)
  kube::test::if_has_string "${output_message}" "alias \"getn\" already exists, use --overwrite to replace"

  # Test: Error cases - Missing required flags
  output_message=$(! kubectl alpha kuberc set --kuberc="$KUBERC_FILE" --command=get --option=output=wide 2>&1)
  kube::test::if_has_string "${output_message}" "required flag(s) \"section\" not set"

  output_message=$(! kubectl alpha kuberc set --kuberc="$KUBERC_FILE" --section=defaults --option=output=wide 2>&1)
  kube::test::if_has_string "${output_message}" "required flag(s) \"command\" not set"

  # Test: KUBERC=off with view command
  output_message=$(! KUBERC=off kubectl alpha kuberc view 2>&1)
  kube::test::if_has_string "${output_message}" "KUBERC is disabled via KUBERC=off environment variable"

  # Test: KUBERC=off with set command
  output_message=$(! KUBERC=off kubectl alpha kuberc set --section=defaults --command=get --option=output=wide 2>&1)
  kube::test::if_has_string "${output_message}" "KUBERC is disabled via KUBERC=off environment variable"

  # Restore getn alias back to "namespace" for remaining tests
  kubectl alpha kuberc set --kuberc="$KUBERC_FILE" --section=aliases --name=getn --command=get --prependarg=namespace --option=output=wide --overwrite
  # Restore get defaults back to namespace=test-kuberc-ns and output=json for remaining tests
  kubectl alpha kuberc set --kuberc="$KUBERC_FILE" --section=defaults --command=get --option=namespace=test-kuberc-ns --option=output=json --overwrite

  kube::log::status "Testing kuberc aliases and defaults functionality"

  # Pre-condition: the test-kuberc-ns namespace does not exist
  kube::test::get_object_assert 'namespaces' "{{range.items}}{{ if eq ${id_field:?} \"test-kuberc-ns\" }}found{{end}}{{end}}:" ':'
  # Alias command crns successfully creates namespace
  kubectl crns --kuberc="$KUBERC_FILE"
  # Post-condition: namespace 'test-kuberc-ns' is created.
  kube::test::get_object_assert 'namespaces/test-kuberc-ns' "{{$id_field}}" 'test-kuberc-ns'

  # Alias command crns successfully creates namespace
  kubectl getn --kuberc="$KUBERC_FILE" test-kuberc-ns
  # Post-condition: namespace 'test-kuberc-ns' is created.
  kube::test::get_object_assert 'namespaces/test-kuberc-ns' "{{$id_field}}" 'test-kuberc-ns'

  # Alias command crns successfully creates namespace
  kubectl getn test-kuberc-ns --output=json --kuberc="$KUBERC_FILE"
  # Post-condition: namespace 'test-kuberc-ns' is created.
  kube::test::get_object_assert 'namespaces/test-kuberc-ns' "{{$id_field}}" 'test-kuberc-ns'

  # check array flags are appended after implicit defaults
  kubectl crole testkubercrole --verb=list --namespace test-kuberc-ns --resource=pods --kuberc="$KUBERC_FILE"
  output_message=$(kubectl getrole role/testkubercrole -n test-kuberc-ns -oyaml --kuberc="$KUBERC_FILE")
  kube::test::if_has_string "${output_message}" 'list'
  kube::test::if_has_not_string "${output_message}" 'watch' 'get'
  # Post-condition: remove role
  kubectl delete role testkubercrole --namespace=test-kuberc-ns

  # Alias run command creates a pod with the given configurations
  kubectl runx --kuberc "$KUBERC_FILE"
  # Post-Condition: assertion object exists
  kube::test::get_object_assert 'pod/test-pod-2 --namespace=test-kuberc-ns' "{{$id_field}}" 'test-pod-2'
  # Not explicitly pass namespace to assure that default flag value is used
  output_message=$(kubectl get pod/test-pod-2 2>&1 "${kube_flags[@]:?}" --kuberc="$KUBERC_FILE")
  kube::test::if_has_string "${output_message}" 'nginx' 'app=test' 'env=test' 'DNS_DOMAIN=test' 'custom-arg1'
  # output flag is defaulted to json and assure that it is correct format
  kube::test::if_has_string "${output_message}" '{'

  # pass explicit invalid namespace to assure that it takes precedence over the value in kuberc
  output_message=$(! kubectl get pod/test-pod-2 -n kube-system 2>&1 "${kube_flags[@]:?}" --kuberc="$KUBERC_FILE")
  kube::test::if_has_string "${output_message}" 'pods "test-pod-2" not found'

  # Alias set env command sets new env var
  kubectl setx --kuberc="$KUBERC_FILE" -n test-kuberc-ns
  # explicitly pass same namespace also defined in kuberc
  output_message=$(kubectl get pod/test-pod-2 -n test-kuberc-ns 2>&1 "${kube_flags[@]:?}" --kuberc="$KUBERC_FILE")
  kube::test::if_has_string "${output_message}" 'busybox'
  kube::test::if_has_not_string "${output_message}" 'nginx'

  # default overrides should prevent actual apply as they are all dry-run=server
  # also assure that explicit flags are also passed
  output_message=$(kubectl apply -n test-kuberc-ns -f hack/testdata/pod.yaml --kuberc="$KUBERC_FILE")
  kube::test::if_has_string "${output_message}" 'serverside-applied (server dry run)'

  # interactive flag is defaulted to true and prompted as no
  output_message=$(kubectl delete pod/test-pod-2 -n test-kuberc-ns <<< $'n\n' --kuberc="$KUBERC_FILE")
  kube::test::if_has_string "${output_message}" 'pod/test-pod-2'
  # assure that it is not deleted
  output_message=$(kubectl get pod/test-pod-2 2>&1 "${kube_flags[@]:?}" --kuberc="$KUBERC_FILE")
  kube::test::if_has_string "${output_message}" "test-pod-2"

  # verify getn alias is working or not, depending if KUBECTL_KUBERC is on or off
  output_message=$(! KUBECTL_KUBERC=false kubectl getn 2>&1 "${kube_flags[@]:?}" --kuberc="$KUBERC_FILE")
  kube::test::if_has_string "${output_message}" "error: unknown command \"getn\" for \"kubectl\""
  KUBECTL_KUBERC=true kubectl getn "${kube_flags[@]:?}" --kuberc="$KUBERC_FILE"

  # verify KUBERC=off is working as expected
  output_message=$(! KUBERC=off kubectl getn 2>&1 "${kube_flags[@]:?}" --kuberc="$KUBERC_FILE")
  kube::test::if_has_string "${output_message}" "KUBERC=off and passing kuberc flag are mutually exclusive"
  output_message=$(! KUBERC=off kubectl getn 2>&1 "${kube_flags[@]:?}")
  kube::test::if_has_string "${output_message}" "error: unknown command \"getn\" for \"kubectl\""

  cat > "${TMPDIR:-/tmp}"/kuberc_file_multi << EOF
---
apiVersion: kubectl.config.k8s.io/v1beta1
kind: Preference
defaults:
- command: get
  options:
  - name: namespace
    default: "test-kuberc-ns"
  - name: output
    default: "json"
unknown: invalid
---
apiVersion: kubectl.config.k8s.io/notexist
kind: Preference
overrides:
- command: get
  flags:
  - name: namespace
    default: "test-kuberc-ns"
  - name: output
    default: "json"
EOF

  # assure that it is not deleted
  output_message=$(kubectl get pod/test-pod-2 2>&1 "${kube_flags[@]:?}" --kuberc="${TMPDIR:-/tmp}"/kuberc_file_multi)
  # assure that correct kuberc is found and printed in output_message
  kube::test::if_has_string "${output_message}" "test-pod-2"
  # assure that warning message is also printed for the notexist kuberc version
  kube::test::if_has_string "${output_message}" "strict decoding error" "unknown"

  touch "${TMPDIR:-/tmp}"/empty_kuberc_file
  output_message=$(kubectl get namespace test-kuberc-ns 2>&1 "${kube_flags[@]:?}" --kuberc="${TMPDIR:-/tmp}"/empty_kuberc_file)
  kube::test::if_has_not_string "${output_message}" "kuberc: no preferences found in"
  output_message=$(kubectl get namespace test-kuberc-ns 2>&1 "${kube_flags[@]:?}" --kuberc="${TMPDIR:-/tmp}"/empty_kuberc_file -v=5)
  kube::test::if_has_string "${output_message}" "kuberc: no preferences found in"
  output_message=$(kubectl get namespace test-kuberc-ns 2>&1 -v 5 "${kube_flags[@]:?}" --kuberc="${TMPDIR:-/tmp}"/empty_kuberc_file)
  kube::test::if_has_string "${output_message}" "kuberc: no preferences found in"
  output_message=$(kubectl get namespace test-kuberc-ns 2>&1 "${kube_flags[@]:?}" --v=5 --kuberc="${TMPDIR:-/tmp}"/empty_kuberc_file)
  kube::test::if_has_string "${output_message}" "kuberc: no preferences found in"
  output_message=$(kubectl get --v 5 namespace test-kuberc-ns 2>&1 "${kube_flags[@]:?}" --kuberc="${TMPDIR:-/tmp}"/empty_kuberc_file)
  kube::test::if_has_string "${output_message}" "kuberc: no preferences found in"

  # explicitly overwriting the value that is also defaulted in kuberc and
  # assure that explicit value supersedes
  output_message=$(kubectl delete namespace/test-kuberc-ns --interactive=false --kuberc="$KUBERC_FILE")
  kube::test::if_has_string "${output_message}" 'namespace "test-kuberc-ns" deleted'

  rm "${TMPDIR:-/tmp}"/kuberc_file

  set +o nounset
  set +o errexit
}
