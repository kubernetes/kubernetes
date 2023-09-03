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

run_job_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing job"

  ### Create a new namespace
  # Pre-condition: the test-jobs namespace does not exist
  kube::test::get_object_assert 'namespaces' "{{range.items}}{{ if eq ${id_field:?} \"test-jobs\" }}found{{end}}{{end}}:" ':'
  # Command
  kubectl create namespace test-jobs
  # Post-condition: namespace 'test-jobs' is created.
  kube::test::get_object_assert 'namespaces/test-jobs' "{{$id_field}}" 'test-jobs'

  # Pre-condition: cronjob does not exist
  kube::test::get_object_assert 'cronjob --namespace=test-jobs' "{{range.items}}{{ if eq $id_field \"pi\" }}found{{end}}{{end}}:" ':'
  # Dry-run create CronJob
  kubectl create cronjob pi --dry-run=client --schedule="59 23 31 2 *" --namespace=test-jobs "--image=$IMAGE_PERL" -- perl -Mbignum=bpi -wle 'print bpi(20)' "${kube_flags[@]:?}"
  kubectl create cronjob pi --dry-run=server --schedule="59 23 31 2 *" --namespace=test-jobs "--image=$IMAGE_PERL" -- perl -Mbignum=bpi -wle 'print bpi(20)' "${kube_flags[@]:?}"
  kube::test::get_object_assert 'cronjob' "{{range.items}}{{ if eq $id_field \"pi\" }}found{{end}}{{end}}:" ':'
  ### Create a cronjob in a specific namespace
  kubectl create cronjob pi --schedule="59 23 31 2 *" --namespace=test-jobs "--image=$IMAGE_PERL" -- perl -Mbignum=bpi -wle 'print bpi(20)' "${kube_flags[@]:?}"
  # Post-Condition: assertion object exists
  kube::test::get_object_assert 'cronjob/pi --namespace=test-jobs' "{{$id_field}}" 'pi'
  kubectl get cronjob/pi --namespace=test-jobs
  kubectl describe cronjob/pi --namespace=test-jobs
  # Describe command should respect the chunk size parameter
  kube::test::describe_resource_chunk_size_assert cronjobs events "--namespace=test-jobs"

  ### Create a job in dry-run mode
  output_message=$(kubectl create job test-job --from=cronjob/pi --dry-run=true --namespace=test-jobs -o name)
  # Post-condition: The text 'job.batch/test-job' should be part of the output
  kube::test::if_has_string "${output_message}" 'job.batch/test-job'
  # Post-condition: The test-job wasn't created actually
  kube::test::get_object_assert jobs "{{range.items}}{{$id_field}}{{end}}" ''

  # Pre-condition: job does not exist
  kube::test::get_object_assert 'job --namespace=test-jobs' "{{range.items}}{{ if eq $id_field \"test-jobs\" }}found{{end}}{{end}}:" ':'
  ### Dry-run create a job in a specific namespace
  kubectl create job test-job --from=cronjob/pi --namespace=test-jobs --dry-run=client
  kubectl create job test-job --from=cronjob/pi --namespace=test-jobs --dry-run=server
  kube::test::get_object_assert 'job --namespace=test-jobs' "{{range.items}}{{ if eq $id_field \"test-jobs\" }}found{{end}}{{end}}:" ':'
  ### Create a job in a specific namespace
  kubectl create job test-job --from=cronjob/pi --namespace=test-jobs
  # Post-Condition: assertion object exists
  kube::test::get_object_assert 'job/test-job --namespace=test-jobs' "{{$id_field}}" 'test-job'
  kubectl get job/test-job --namespace=test-jobs
  kubectl describe job/test-job --namespace=test-jobs
  # Describe command should respect the chunk size parameter
  kube::test::describe_resource_chunk_size_assert jobs events "--namespace=test-jobs"
  #Clean up
  kubectl delete job test-job --namespace=test-jobs
  kubectl delete cronjob pi --namespace=test-jobs
  kubectl delete namespace test-jobs

  set +o nounset
  set +o errexit
}
