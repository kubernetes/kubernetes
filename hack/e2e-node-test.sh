#!/bin/bash

# Copyright 2016 The Kubernetes Authors All rights reserved.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

focus=${FOCUS:-""}
skip=${SKIP:-""}
report=${REPORT:-"/tmp/"}
artifacts=${ARTIFACTS:-"/tmp/_artifacts"}
remote=${REMOTE:-"false"}
images=${IMAGES:-""}
hosts=${HOSTS:-""}
if [[ $hosts == "" && $images == "" ]]; then
  images="e2e-node-containervm-v20160321-image"
fi
image_project=${IMAGE_PROJECT:-"kubernetes-node-e2e-images"}
instance_prefix=${INSTANCE_PREFIX:-"test"}
cleanup=${CLEANUP:-"true"}
delete_instances=${DELETE_INSTANCES:-"false"}
run_until_failure=${RUN_UNTIL_FAILURE:-"false"}
list_images=${LIST_IMAGES:-"false"}

if  [[ $list_images == "true" ]]; then
  gcloud compute images list --project="${image_project}" | grep "e2e-node"
  exit 0
fi

ginkgo=$(kube::util::find-binary "ginkgo")
if [[ -z "${ginkgo}" ]]; then
  echo "You do not appear to have ginkgo built. Try 'make WHAT=vendor/github.com/onsi/ginkgo/ginkgo'"
  exit 1
fi

if [ $remote = true ] ; then
  # Setup the directory to copy test artifacts (logs, junit.xml, etc) from remote host to local host
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

  # Parse the flags to pass to ginkgo
  ginkgoflags=""
  if [[ $focus != "" ]]; then
     ginkgoflags="$ginkgoflags -focus=$focus "
  fi

  if [[ $skip != "" ]]; then
     ginkgoflags="$ginkgoflags -skip=$skip "
  fi

  if [[ $run_until_failure != "" ]]; then
     ginkgoflags="$ginkgoflags -untilItFails=$run_until_failure "
  fi

  # Output the configuration we will try to run
  echo "Running tests remotely using"
  echo "Project: $project"
  echo "Image Project: $image_project"
  echo "Compute/Zone: $zone"
  echo "Images: $images"
  echo "Hosts: $hosts"
  echo "Ginkgo Flags: $ginkgoflags"

  # Invoke the runner
  go run test/e2e_node/runner/run_e2e.go  --logtostderr --vmodule=*=2 --ssh-env="gce" \
    --zone="$zone" --project="$project"  \
    --hosts="$hosts" --images="$images" --cleanup="$cleanup" \
    --results-dir="$artifacts" --ginkgo-flags="$ginkgoflags" \
    --image-project="$image_project" --instance-name-prefix="$instance_prefix" --setup-node="true" \
    --delete-instances="$delete_instances"
  exit $?

else
  # Refresh sudo credentials if not running on GCE.
  if ! ping -c 1 -q metadata.google.internal &> /dev/null; then
    sudo -v || exit 1
  fi

  # Test using the host the script was run on
  # Provided for backwards compatibility
  "${ginkgo}" --focus=$focus --skip=$skip "${KUBE_ROOT}/test/e2e_node/" --report-dir=${report} \
    -- --alsologtostderr --v 2 --node-name $(hostname) --build-services=true --start-services=true --stop-services=true
  exit $?
fi
