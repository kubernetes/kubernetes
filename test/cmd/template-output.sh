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

run_template_output_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing --template support on commands"
  ### Test global request timeout option
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  # check that create supports --template output
  kubectl create "${kube_flags[@]}" -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml
  # Post-condition: valid-pod POD is created
  kubectl get "${kube_flags[@]}" pods -o json
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  # check that patch command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" patch --dry-run pods/valid-pod -p '{"patched":"value3"}' --type=merge --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'valid-pod:'

  # check that label command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" label --dry-run pods/valid-pod label=value --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'valid-pod:'

  # check that annotate command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" annotate --dry-run pods/valid-pod annotation=value --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'valid-pod:'

  # check that apply command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" apply --dry-run -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'valid-pod:'

  # check that create command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml --dry-run --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'valid-pod:'

  # check that autoscale command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" autoscale --max=2 -f hack/testdata/scale-deploy-1.yaml --dry-run --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'scale-1:'

  # check that expose command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" expose -f hack/testdata/redis-slave-replicaset.yaml --save-config --port=80 --target-port=8000 --dry-run --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'redis-slave:'

  # check that convert command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" convert -f hack/testdata/deployment-revision1.yaml --output-version=apps/v1beta1 --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'nginx:'

  # check that run command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" run --dry-run --template="{{ .metadata.name }}:" pi --image=perl --restart=OnFailure -- perl -Mbignum=bpi -wle 'print bpi(2000)')
  kube::test::if_has_string "${output_message}" 'pi:'

  # check that taint command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" taint node 127.0.0.1 dedicated=foo:PreferNoSchedule --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" '127.0.0.1:'
  # untaint node
  kubectl taint node 127.0.0.1 dedicated-

  # check that "apply set-last-applied" command supports --template output
  kubectl "${kube_flags[@]}" create -f test/e2e/testing-manifests/statefulset/cassandra/controller.yaml
  output_message=$(kubectl "${kube_flags[@]}" apply set-last-applied -f test/e2e/testing-manifests/statefulset/cassandra/controller.yaml --dry-run --create-annotation --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'cassandra:'

  # check that "auth reconcile" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" auth reconcile --dry-run -f test/fixtures/pkg/kubectl/cmd/auth/rbac-resource-plus.yaml --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'testing-CR:testing-CRB:testing-RB:testing-R:'

  # check that "create clusterrole" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" create clusterrole --template="{{ .metadata.name }}:" --verb get myclusterrole  --non-resource-url /logs/ --resource pods)
  kube::test::if_has_string "${output_message}" 'myclusterrole:'

  # check that "create clusterrolebinding" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" create clusterrolebinding foo --clusterrole=myclusterrole --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'foo:'

  # check that "create configmap" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" create configmap cm --dry-run --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'cm:'

  # check that "create deployment" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" create deployment deploy --image=nginx --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'deploy:'

  # check that "create job" command supports --template output
  kubectl create "${kube_flags[@]}" -f - <<EOF
apiVersion: batch/v1beta1
kind: CronJob
metadata:
  name: pi
spec:
  schedule: "*/10 * * * *"
  jobTemplate:
    spec:
      template:
        metadata:
          labels:
            parent: "pi"
        spec:
          containers:
          - name: pi
            image: perl
            command: ["perl",  "-Mbignum=bpi", "-wle", "print bpi(2000)"]
          restartPolicy: OnFailure
EOF
  output_message=$(kubectl "${kube_flags[@]}" create job foo --from=cronjob/pi --dry-run --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'foo:'

  # check that "create namespace" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" create ns bar --dry-run --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'bar:'

  # check that "create namespace" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" create rolebinding foo --clusterrole=myclusterrole --dry-run --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'foo:'

  # check that "create role" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" create role --dry-run --template="{{ .metadata.name }}:" --verb get myrole --resource pods)
  kube::test::if_has_string "${output_message}" 'myrole:'

  # check that "create quota" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" create quota foo --dry-run --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'foo:'

  # check that "create priorityclass" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" create priorityclass foo --dry-run --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'foo:'

  # check that "create poddisruptionbudget" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" create poddisruptionbudget foo --dry-run --selector=foo --min-available=1 --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'foo:'

  # check that "create serviceaccount" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" create serviceaccount foo --dry-run --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'foo:'

  # check that "set env" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" set env pod/valid-pod --dry-run A=B --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'valid-pod:'

  # check that "set image" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" set image pod/valid-pod --dry-run kubernetes-serve-hostname=nginx --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'valid-pod:'

  # check that "set resources" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" set resources pod/valid-pod --limits=memory=256Mi --dry-run --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'valid-pod:'

  # check that "set selector" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" set selector -f hack/testdata/kubernetes-service.yaml A=B --local --dry-run --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'kubernetes:'

  # check that "set serviceaccount" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" set serviceaccount pod/valid-pod deployer --dry-run --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'valid-pod:'

  # check that "set subject" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" set subject clusterrolebinding/foo --user=foo --dry-run --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'foo:'

  # check that "create secret docker-registry" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" create secret docker-registry foo --docker-username user --docker-password pass --docker-email foo@bar.baz --dry-run --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'foo:'

  # check that "create secret generic" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" create secret generic foo --from-literal=key1=value1 --dry-run --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'foo:'

  # check that "create secret tls" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" create secret tls --dry-run foo --key=hack/testdata/tls.key --cert=hack/testdata/tls.crt --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'foo:'

  # check that "create service clusterip" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" create service clusterip foo --dry-run --tcp=8080 --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'foo:'

  # check that "create service externalname" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" create service externalname foo --dry-run --external-name=bar --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'foo:'

  # check that "create service loadbalancer" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" create service loadbalancer foo --dry-run --tcp=8080 --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'foo:'

  # check that "create service nodeport" command supports --template output
  output_message=$(kubectl "${kube_flags[@]}" create service nodeport foo --dry-run --tcp=8080 --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'foo:'

  # check that "config view" ouputs "yaml" as its default output format
  output_message=$(kubectl "${kube_flags[@]}" config view)
  kube::test::if_has_string "${output_message}" 'kind: Config'

  # check that "rollout pause" supports --template output
  output_message=$(kubectl "${kube_flags[@]}" rollout pause deploy/deploy --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'deploy:'

   # check that "rollout history" supports --template output
  output_message=$(kubectl "${kube_flags[@]}" rollout history deploy/deploy --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'deploy:'

  # check that "rollout resume" supports --template output
  output_message=$(kubectl "${kube_flags[@]}" rollout resume deploy/deploy --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'deploy:'

  # check that "rollout undo" supports --template output
  output_message=$(kubectl "${kube_flags[@]}" rollout undo deploy/deploy --to-revision=1 --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'deploy:'

  # check that "config view" command supports --template output
  # and that commands that set a default output (yaml in this case),
  # default to "go-template" as their output format when a --template
  # value is provided, but no explicit --output format is given.
  output_message=$(kubectl "${kube_flags[@]}" config view --template="{{ .kind }}:")
  kube::test::if_has_string "${output_message}" 'Config'

  # check that running a command with both a --template flag and a
  # non-template --output prefers the non-template output value
  output_message=$(kubectl "${kube_flags[@]}" create configmap cm --dry-run --template="{{ .metadata.name }}:" --output yaml)
  kube::test::if_has_string "${output_message}" 'kind: ConfigMap'

  # cleanup
  kubectl delete cronjob pi "${kube_flags[@]}"
  kubectl delete pods --all "${kube_flags[@]}"
  kubectl delete rc cassandra "${kube_flags[@]}"
  kubectl delete clusterrole myclusterrole "${kube_flags[@]}"
  kubectl delete clusterrolebinding foo "${kube_flags[@]}"
  kubectl delete deployment deploy "${kube_flags[@]}"

  set +o nounset
  set +o errexit
}
