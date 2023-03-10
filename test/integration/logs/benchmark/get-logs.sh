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

# Usage: get-logs.sh [<job ID>]
#
# Downloads the latest job output or the one with the specified ID
# and prepares running benchmarks for it.

set -o pipefail
set -o errexit
set -x

cd "$(dirname "$0")"

latest_job () {
    gsutil cat gs://kubernetes-jenkins/logs/ci-kubernetes-kind-e2e-json-logging/latest-build.txt
}

job=${1:-$(latest_job)}

rm -rf ci-kubernetes-kind-e2e-json-logging
mkdir ci-kubernetes-kind-e2e-json-logging
gsutil -m cp -R "gs://kubernetes-jenkins/logs/ci-kubernetes-kind-e2e-json-logging/${job}/*" ci-kubernetes-kind-e2e-json-logging/

for i in kube-apiserver kube-controller-manager kube-scheduler; do
    # Before (container runtime log dump (?)):
    #   2023-03-07T07:30:52.193301924Z stderr F {"ts":1678174252192.0676,"caller":"scheduler/schedule_one.go:81","msg":"Attempting to schedule pod","v":3,"pod":{"name":"simpletest.rc-zgd47","namespace":"gc-5422"}}
    # After:
    #   {"ts":1678174252192.0676,"caller":"scheduler/schedule_one.go:81","msg":"Attempting to schedule pod","v":3,"pod":{"name":"simpletest.rc-zgd47","namespace":"gc-5422"}}
    sed -e 's/^20[^ ]* stderr . //' \
        ci-kubernetes-kind-e2e-json-logging/artifacts/kind-control-plane/containers/$i-*.log \
        > ci-kubernetes-kind-e2e-json-logging/$i.log;
done

# Before (systemd format):
#   Mar 07 07:22:05 kind-control-plane kubelet[288]: {"ts":1678173725722.4487,"caller":"flag/flags.go:64","msg":"FLAG: --address=\"0.0.0.0\"\n","v":1}
# After:
#   {"ts":1678173725722.4487,"caller":"flag/flags.go:64","msg":"FLAG: --address=\"0.0.0.0\"\n","v":1}
grep 'kind-worker kubelet' ci-kubernetes-kind-e2e-json-logging/artifacts/kind-worker/kubelet.log | \
    sed -e 's;^.* kind-worker kubelet[^ ]*: ;;' > ci-kubernetes-kind-e2e-json-logging/kind-worker-kubelet.log

# Create copies of the actual files, whether they already exist or not. To
# clean up disk space, use "git clean -fx test/integration/logs/benchmark".
copy () {
    from="$1"
    to="$2"

    mkdir -p "$(dirname "$to")"
    rm -f "$to"
    cp "$from" "$to"
}
copy ci-kubernetes-kind-e2e-json-logging/kind-worker-kubelet.log data/kind-worker-kubelet.log
copy ci-kubernetes-kind-e2e-json-logging/kube-apiserver.log data/kube-apiserver.log
copy ci-kubernetes-kind-e2e-json-logging/kube-controller-manager.log data/kube-controller-manager.log
copy ci-kubernetes-kind-e2e-json-logging/kube-scheduler.log data/kube-scheduler.log

copy ci-kubernetes-kind-e2e-json-logging/kind-worker-kubelet.log data/v3/kind-worker-kubelet.log
copy ci-kubernetes-kind-e2e-json-logging/kube-apiserver.log data/v3/kube-apiserver.log
copy ci-kubernetes-kind-e2e-json-logging/kube-controller-manager.log data/v3/kube-controller-manager.log
copy ci-kubernetes-kind-e2e-json-logging/kube-scheduler.log data/v3/kube-scheduler.log
