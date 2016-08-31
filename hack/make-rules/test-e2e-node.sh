#!/bin/bash

# Copyright 2016 The Kubernetes Authors.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/hack/lib/init.sh"

focus=${FOCUS:-""}
skip=${SKIP:-""}
# The number of tests that can run in parallel depends on what tests
# are running and on the size of the node. Too many, and tests will
# fail due to resource contention. 8 is a reasonable default for a
# n1-standard-1 node.
# Currently, parallelism only affects when REMOTE=true. For local test,
# ginkgo default parallelism (cores - 1) is used.
parallelism=${PARALLELISM:-8}
report=${REPORT:-"/tmp/"}
artifacts=${ARTIFACTS:-"/tmp/_artifacts"}
remote=${REMOTE:-"false"}
run_until_failure=${RUN_UNTIL_FAILURE:-"false"}
test_args=${TEST_ARGS:-""}

# Parse the flags to pass to ginkgo
ginkgoflags=""
if [[ $parallelism > 1 ]]; then
  ginkgoflags="$ginkgoflags -nodes=$parallelism "
fi

if [[ $focus != "" ]]; then
  ginkgoflags="$ginkgoflags -focus=$focus "
fi

if [[ $skip != "" ]]; then
  ginkgoflags="$ginkgoflags -skip=$skip "
fi

if [[ $run_until_failure != "" ]]; then
  ginkgoflags="$ginkgoflags -untilItFails=$run_until_failure "
fi


if [ $remote = true ] ; then
  # The following options are only valid in remote run.
  images=${IMAGES:-""}
  hosts=${HOSTS:-""}
  image_project=${IMAGE_PROJECT:-"kubernetes-node-e2e-images"}
  metadata=${INSTANCE_METADATA:-""}
  list_images=${LIST_IMAGES:-false}
  if  [[ $list_images == "true" ]]; then
    gcloud compute images list --project="${image_project}" | grep "e2e-node"
    exit 0
  fi
  gubernator=${GUBERNATOR:-"false"}
  if [[ $hosts == "" && $images == "" ]]; then
    image_project=${IMAGE_PROJECT:-"google-containers"}
    gci_image=$(gcloud compute images list --project $image_project \
    --no-standard-images --regexp="gci-dev.*" --format="table[no-heading](name)")
    images=$gci_image
    metadata="user-data<${KUBE_ROOT}/test/e2e_node/jenkins/gci-init.yaml"
  fi
  instance_prefix=${INSTANCE_PREFIX:-"test"}
  cleanup=${CLEANUP:-"true"}
  delete_instances=${DELETE_INSTANCES:-"false"}

  # Setup the directory to copy test artifacts (logs, junit.xml, etc) from remote host to local host
  if [[ $gubernator = true && -d "${artifacts}" ]]; then
    echo "Removing artifacts directory at ${artifacts}"
    rm -r ${artifacts}
  fi

  if [ ! -d "${artifacts}" ]; then
    echo "Creating artifacts directory at ${artifacts}"
    mkdir -p ${artifacts}
  fi
  echo "Test artifacts will be written to ${artifacts}"

  # Get the compute zone
  zone=$(gcloud info --format='value(config.properties.compute.zone)')
  if [[ $zone == "" ]]; then
    echo "Could not find gcloud compute/zone when running:\ngcloud info --format='value(config.properties.compute.zone)'"
    exit 1
  fi

  # Get the compute project
  project=$(gcloud info --format='value(config.project)')
  if [[ $project == "" ]]; then
    echo "Could not find gcloud project when running:\ngcloud info --format='value(config.project)'"
    exit 1
  fi

  # Check if any of the images specified already have running instances.  If so reuse those instances
  # by moving the IMAGE to a HOST
  if [[ $images != "" ]]; then
  IFS=',' read -ra IM <<< "$images"
       images=""
       for i in "${IM[@]}"; do
         if [[ $(gcloud compute instances list "${instance_prefix}-$i" | grep $i) ]]; then
           if [[ $hosts != "" ]]; then
             hosts="$hosts,"
           fi
           echo "Reusing host ${instance_prefix}-$i"
           hosts="${hosts}${instance_prefix}-${i}"
         else
           if [[ $images != "" ]]; then
             images="$images,"
           fi
           images="$images$i"
         fi
       done
  fi

  # Output the configuration we will try to run
  echo "Running tests remotely using"
  echo "Project: $project"
  echo "Image Project: $image_project"
  echo "Compute/Zone: $zone"
  echo "Images: $images"
  echo "Hosts: $hosts"
  echo "Ginkgo Flags: $ginkgoflags"
  echo "Instance Metadata: $metadata"
  # Invoke the runner
  go run test/e2e_node/runner/remote/run_remote.go  --logtostderr --vmodule=*=4 --ssh-env="gce" \
    --zone="$zone" --project="$project" --gubernator="$gubernator" \
    --hosts="$hosts" --images="$images" --cleanup="$cleanup" \
    --results-dir="$artifacts" --ginkgo-flags="$ginkgoflags" \
    --image-project="$image_project" --instance-name-prefix="$instance_prefix" --setup-node="true" \
    --delete-instances="$delete_instances" --test_args="$test_args" --instance-metadata="$metadata" \
    2>&1 | tee "${artifacts}/build-log.txt"
  exit $?

else
  # Refresh sudo credentials if not running on GCE.
  if ! ping -c 1 -q metadata.google.internal &> /dev/null; then
    sudo -v || exit 1
  fi

  # If the flag --disable-kubenet is not set, set true by default.
  if ! [[ $test_args =~ "--disable-kubenet" ]]; then
    test_args="$test_args --disable-kubenet=true"
  fi

  # On selinux enabled systems, it might
  # require to relabel /var/lib/kubelet
  if which selinuxenabled &> /dev/null && \
     selinuxenabled && \
     which chcon > /dev/null ; then
     mkdir -p /var/lib/kubelet
     if [[ ! $(ls -Zd /var/lib/kubelet) =~ svirt_sandbox_file_t ]] ; then
        echo "Applying SELinux label to /var/lib/kubelet directory."
        if ! sudo chcon -Rt svirt_sandbox_file_t /var/lib/kubelet; then
           echo "Failed to apply selinux label to /var/lib/kubelet."
        fi
     fi
  fi

  # Test using the host the script was run on
  # Provided for backwards compatibility
  go run test/e2e_node/runner/local/run_local.go --ginkgo-flags="$ginkgoflags" \
    --test-flags="--alsologtostderr --v 4 --report-dir=${report} --node-name $(hostname) \
    $test_args" --build-dependencies=true
  exit $?
fi
