#!/usr/bin/env bash

# Copyright 2022 The Kubernetes Authors.
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

# Runs tests for kubectl diff
run_kubectl_events_tests() {
    set -o nounset
    set -o errexit

    create_and_use_new_namespace
    kube::log::status "Testing kubectl events"

    ### Create a new namespace
    # Pre-condition: the test-events namespace does not exist
    kube::test::get_object_assert 'namespaces' "{{range.items}}{{ if eq ${id_field:?} \\\"test-events\\\" }}found{{end}}{{end}}:" ':'
    # Command
    kubectl create namespace test-events
    # Post-condition: namespace 'test-events' is created.
    kube::test::get_object_assert 'namespaces/test-events' "{{$id_field}}" 'test-events'

    # Pre-condition: event does not exist for Cronjob/pi in any namespace
    output_message=$(kubectl alpha events -A "${kube_flags[@]:?}" 2>&1)
    kube::test::if_has_not_string "${output_message}" "Warning" "InvalidSchedule" "Cronjob/pi"

    # Pre-condition: cronjob does not exist in test-events namespace
    kube::test::get_object_assert 'cronjob --namespace=test-events' "{{range.items}}{{ if eq $id_field \\\"pi\\\" }}found{{end}}{{end}}:" ':'
    ### Create a cronjob in a specific namespace
    kubectl create cronjob pi --schedule="59 23 31 2 *" --namespace=test-events "--image=$IMAGE_PERL" -- perl -Mbignum=bpi -wle 'print bpi(20)' "${kube_flags[@]:?}"
    # Post-Condition: assertion object exists
    kube::test::get_object_assert 'cronjob/pi --namespace=test-events' "{{$id_field}}" 'pi'

    # Post-Condition: events --all-namespaces returns event for Cronjob/pi
    output_message=$(kubectl alpha events -A "${kube_flags[@]:?}" 2>&1)
    kube::test::if_has_string "${output_message}" "Warning" "InvalidSchedule" "Cronjob/pi"

    # Post-Condition: events for test-events namespace returns event for Cronjob/pi
    output_message=$(kubectl alpha events -n test-events "${kube_flags[@]:?}" 2>&1)
    kube::test::if_has_string "${output_message}" "Warning" "InvalidSchedule" "Cronjob/pi"

    # Post-Condition: events returns event for Cronjob/pi when --for flag is used
    output_message=$(kubectl alpha events -n test-events --for=Cronjob/pi "${kube_flags[@]:?}" 2>&1)
    kube::test::if_has_string "${output_message}" "Warning" "InvalidSchedule" "Cronjob/pi"

    # Post-Condition: events returns event for Cronjob/pi when watch is enabled
    output_message=$(kubectl alpha events -n test-events --for=Cronjob/pi --watch --request-timeout=1 "${kube_flags[@]:?}" 2>&1)
    kube::test::if_has_string "${output_message}" "Warning" "InvalidSchedule" "Cronjob/pi"

    #Clean up
    kubectl delete cronjob pi --namespace=test-events
    kubectl delete namespace test-events

    set +o nounset
    set +o errexit
}
