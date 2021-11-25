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

# This script is used to avoid regressions after a package is migrated
# to structured logging. once a package is completely migrated add
# it .structured_logging file to avoid any future regressions.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

kube::golang::verify_go_version

# Ensure that we find the binaries we build before anything else.
export GOBIN="${KUBE_OUTPUT_BINPATH}"
PATH="${GOBIN}:${PATH}"

# Install logcheck
pushd "${KUBE_ROOT}/hack/tools" >/dev/null
  GO111MODULE=on go install k8s.io/klog/hack/tools/logcheck
popd >/dev/null

# We store packages that are migrated in .structured_logging
migrated_packages_file="${KUBE_ROOT}/hack/.structured_logging"

# Ensure that file is sorted.
kube::util::check-file-in-alphabetical-order "${migrated_packages_file}"

migrated_packages=()
while IFS='' read -r line; do
  migrated_packages+=("$KUBE_ROOT/$line")
done < <(cat "${migrated_packages_file}")

echo "Running structured logging static check on migrated packages"
ret=0
GOOS=linux    logcheck "${migrated_packages[@]}" || ret=$?
GOOS=windows  logcheck "${migrated_packages[@]}" || ret=$?

if [ $ret -eq 0 ]; then
  echo "Structured logging static check passed on migrated packages :)"
else
  echo "Please fix above failures. You can locally test via:"
  echo "hack/verify-structured-logging.sh"
fi

# Ignore migrated packages as they are already tested
# Trim the trailing /... from given packages
ignore_packages=$(grep -oE '^[a-zA-Z0-9/]+[^/\.]' < "${KUBE_ROOT}"/hack/.structured_logging)

all_packages=()
# shellcheck disable=SC2010
for i in $(ls -d ./*/ | grep -v 'staging' | grep -v 'vendor' | grep -v 'hack')
do
     all_packages+=("$(go list ./"$i"/... 2> /dev/null | sed 's/k8s.io\/kubernetes\///g')")
  all_packages+=(" ")
done
# We exclude vendor/ except vendor/k8s.io
# This is because vendor/k8s.io is symlinked to staging
# and needs to be checked
all_packages+=("vendor/k8s.io")

# Packages to test = all_packages - ignored_packages
packages=()
while IFS='' read -r line; do
  if [ -z "$line" ]; then continue; fi
  packages+=("$KUBE_ROOT/$line/...")
done < <(echo "${all_packages[@]}" | tr " " "\n" | grep -v "$ignore_packages")

echo -e "\nRunning structured logging static check on all other packages"
GOOS=linux    logcheck -allow-unstructured "${packages[@]}" || ret=$?
GOOS=windows  logcheck -allow-unstructured "${packages[@]}" || ret=$?

if [ $ret -eq 0 ]; then
  echo "Structured logging static check passed on all packages :)"
else
  echo "Please fix above failures. You can locally test via:"
  echo "hack/verify-structured-logging.sh"
fi

exit $ret
