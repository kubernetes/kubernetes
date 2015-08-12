#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

# Updates the docs to be ready to be used as release docs for a particular
# version.
# Example usage:
# ./versionize-docs.sh v1.0.1

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

NEW_VERSION=${1-}

if [ "$#" -lt 1 ]; then
    echo "Usage: versionize-docs <release-version>"
    exit 1
fi

SED=sed
if which gsed &>/dev/null; then
  SED=gsed
fi
if ! ($SED --version 2>&1 | grep -q GNU); then
  echo "!!! GNU sed is required.  If on OS X, use 'brew install gnu-sed'."
  exit 1
fi

echo "+++ Versioning documentation and examples"

# Update the docs to match this version.
HTML_PREVIEW_PREFIX="https://htmlpreview.github.io/\?https://github.com/kubernetes/kubernetes"

md_dirs=(docs examples)
md_files=()
for dir in "${md_dirs[@]}"; do
  mdfiles+=($( find "${dir}" -name "*.md" -type f ))
done
for doc in "${mdfiles[@]}"; do
  $SED -ri \
      -e '/<!-- BEGIN STRIP_FOR_RELEASE -->/,/<!-- END STRIP_FOR_RELEASE -->/d' \
      -e "s|(releases.k8s.io)/[^/]+|\1/${NEW_VERSION}|g" \
      "${doc}"

  # Replace /HEAD in html preview links with /NEW_VERSION.
  $SED -ri -e "s|(${HTML_PREVIEW_PREFIX}/HEAD)|${HTML_PREVIEW_PREFIX}/${NEW_VERSION}|" "${doc}"

  is_versioned_tag="<!-- BEGIN MUNGE: IS_VERSIONED -->
  <!-- TAG IS_VERSIONED -->
  <!-- END MUNGE: IS_VERSIONED -->"
  if ! grep -q "${is_versioned_tag}" "${doc}"; then
    echo -e "\n\n${is_versioned_tag}\n\n" >> "${doc}"
  fi
done

# Update API descriptions to match this version.
$SED -ri -e "s|(releases.k8s.io)/[^/]+|\1/${NEW_VERSION}|" pkg/api/v[0-9]*/types.go

${KUBE_ROOT}/hack/update-generated-docs.sh
${KUBE_ROOT}/hack/update-swagger-spec.sh
