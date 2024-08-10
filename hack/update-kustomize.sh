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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env
kube::util::require-jq
kube::util::ensure_clean_working_dir

PUBLISHED_RELEASES=$(curl -sL 'http://api.github.com/repos/kubernetes-sigs/kustomize/releases?per_page=100' | jq '[ .[] | select(.draft == false and .prerelease == false) | { "tag_name": .tag_name, "published_at": .published_at } ]')

LATEST_KYAML=$(echo "${PUBLISHED_RELEASES}" | jq -r '[ .[] | select(.tag_name | startswith("kyaml/v")) ] | sort_by(.published_at) | last | .tag_name | scan("\/(v[\\d.]+)") | .[0]')
LATEST_CONFIG=$(echo "${PUBLISHED_RELEASES}" | jq -r '[ .[] | select(.tag_name | startswith("cmd/config/v")) ] | sort_by(.published_at) | last | .tag_name | scan("\/(v[\\d.]+)") | .[0]')
LATEST_API=$(echo "${PUBLISHED_RELEASES}" | jq -r '[ .[] | select(.tag_name | startswith("api/v")) ] | sort_by(.published_at) | last | .tag_name | scan("\/(v[\\d.]+)") | .[0]')
LATEST_KUSTOMIZE=$(echo "${PUBLISHED_RELEASES}" | jq -r '[ .[] | select(.tag_name | startswith("kustomize/v")) ] | sort_by(.published_at) | last | .tag_name | scan("\/(v[\\d.]+)") | .[0]')

echo "----------------------------------------"
echo "Latest kyaml: $LATEST_KYAML"
echo "Latest cmd/config: $LATEST_CONFIG"
echo "Latest api: $LATEST_API"
echo "Latest kustomize: $LATEST_KUSTOMIZE"
echo -e "----------------------------------------\n"
read -p "Update kubectl kustomize to these versions? [y/N] " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo -e "\n${color_blue:?}Updating kubectl kustomize${color_norm:?}"
else
  echo -e "\n${color_red:?}Update aborted${color_norm:?}"
  exit 1
fi

./hack/pin-dependency.sh sigs.k8s.io/kustomize/kyaml "$LATEST_KYAML"
./hack/pin-dependency.sh sigs.k8s.io/kustomize/cmd/config "$LATEST_CONFIG"
./hack/pin-dependency.sh sigs.k8s.io/kustomize/api "$LATEST_API"
./hack/pin-dependency.sh sigs.k8s.io/kustomize/kustomize/v5 "$LATEST_KUSTOMIZE"

./hack/update-vendor.sh
./hack/update-internal-modules.sh
./hack/update-go-workspace.sh
./hack/lint-dependencies.sh

sed -i'' -e "s/const kustomizeVersion.*$/const kustomizeVersion = \"${LATEST_KUSTOMIZE}\"/" staging/src/k8s.io/kubectl/pkg/cmd/version/version.go

echo -e "\n${color_blue}Committing changes${color_norm}"
git add .
git commit -a -m "Update kubectl kustomize to kyaml/$LATEST_KYAML, cmd/config/$LATEST_CONFIG, api/$LATEST_API, kustomize/$LATEST_KUSTOMIZE"

echo -e "\n${color_blue:?}Verifying kubectl kustomize version${color_norm:?}"
# We use `make` here instead of `go install` to ensure that all of the
# linker-defined values are set.
make -C "${KUBE_ROOT}" WHAT=./cmd/kubectl

if [[ $(kubectl version --client -o json | jq -r '.kustomizeVersion') != "$LATEST_KUSTOMIZE" ]]; then
  echo -e "${color_red:?}Unexpected kubectl kustomize version${color_norm:?}"
  exit 1
fi

echo -e "\n${color_green:?}Update successful${color_norm:?}"
echo "Note: If any of the integration points changed, you may need to update them manually."
echo "See https://github.com/kubernetes-sigs/kustomize/tree/master/releasing#update-kustomize-in-kubectl for more information"
