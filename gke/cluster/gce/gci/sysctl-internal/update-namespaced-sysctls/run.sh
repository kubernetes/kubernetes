#!/usr/bin/env bash

# Copyright 2021 The Kubernetes Authors.
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

set -Eeuo pipefail

msg() {
  echo >&2 -e "${1-}"
}

function generate_namespaced() {
  local project="${1}"
  local zone="${2}"
  local image_project="${3}"
  local image="${4}"

  local git_root
  git_root=$(git rev-parse --show-toplevel)
  if [[ ${git_root} != $(pwd) ]]; then
    echo "Run this script with your pwd being root of k8s repo"
    exit 1
  fi

  local vm_name
  vm_name="sysctl-test-vm-$(uuidgen)"

  msg "Starting ${vm_name}..."
  gcloud compute instances create "${vm_name}" \
    --image-project="${image_project}" \
    --zone="${zone}" \
    --image="${image}" \
    --project="${project}" \
    --machine-type=e2-medium
  echo "Waiting for ${vm_name} to start..."
  sleep 60

  msg "Copying generate_namespaced.py script to ${vm_name}"
  gcloud compute scp "${git_root}/gke/cluster/gce/gci/systl-internal/update-namespaced-sysctls/generate_namespaced.py" "${vm_name}:/tmp" \
    --project="${project}" \
    --zone="${zone}"

  msg "Running generate_namespaced.py script on ${vm_name}"
  gcloud compute ssh "${vm_name}" \
    --project="${project}" \
    --zone="${zone}" \
    --command="python3 /tmp/generate_namespaced.py --out-file=/tmp/namespaced_sysctsl.yaml"

  local local_out_path="/tmp/namespaced_sysctls_${vm_name}.yaml"
  msg "Copying generate_namespaced.py output to ${local_out_path}"
  gcloud compute scp "${vm_name}:/tmp/namespaced_sysctsl.yaml" "${local_out_path}" \
    --project="${project}" \
    --zone="${zone}"

  msg "Deleting ${vm_name}"
  gcloud compute instances delete "${vm_name}" \
      --project="${project}" \
      --zone="${zone}" \
      --quiet

  msg "All done! Output is at ${local_out_path}"
}

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [-h] [-v] --project project --zone zone --image_project image_project --image image

Generates namespaced sysctls for a given image.

Available options:

-h, --help      Print this help and exit
-v, --verbose   Print script debug info
--project       Project to be used for test VM
--zone          Zone VM will be created in
--image_project Image project to be used
--image         Image name to be used
EOF
  exit
}

die() {
  local msg=$1
  local code=${2-1} # default exit status 1
  msg "$msg"
  exit "$code"
}

parse_params() {
  project=''
  zone=''
  image=''
  image_project=''

  while :; do
    case "${1-}" in
    -h | --help) usage ;;
    -v | --verbose) set -x ;;
    --project)
      project="${2-}"
      shift
      ;;
    --zone)
      zone="${2-}"
      shift
      ;;
    --image_project)
      image_project="${2-}"
      shift
      ;;
    --image)
      image="${2-}"
      shift
      ;;
    -?*) die "Unknown option: $1" ;;
    *) break ;;
    esac
    shift
  done

  # check required params and arguments
  [[ -z "${project-}" ]] && die "Missing required parameter: project"
  [[ -z "${zone-}" ]] && die "Missing required parameter: zone"
  [[ -z "${image_project-}" ]] && die "Missing required parameter: image_project"
  [[ -z "${image-}" ]] && die "Missing required parameter: image"

  generate_namespaced "${project}" "${zone}" "${image_project}" "${image}"

  return 0
}

parse_params "$@"
