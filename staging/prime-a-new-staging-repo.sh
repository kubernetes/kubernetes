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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

dir=$(mktemp -d "${TMPDIR:-/tmp/}$(basename 0).XXXXXXXXXXXX")
echo ${dir}

echo k8s.io/kubernetes/pkg/api/install > ${dir}/packages.txt
echo k8s.io/kubernetes/pkg/apis/apps/install >> ${dir}/packages.txt
echo k8s.io/kubernetes/pkg/apis/authentication/install >> ${dir}/packages.txt
echo k8s.io/kubernetes/pkg/apis/authorization/install >> ${dir}/packages.txt
echo k8s.io/kubernetes/pkg/apis/autoscaling/install >> ${dir}/packages.txt
echo k8s.io/kubernetes/pkg/apis/batch/install >> ${dir}/packages.txt
echo k8s.io/kubernetes/pkg/apis/certificates/install >> ${dir}/packages.txt
echo k8s.io/kubernetes/pkg/apis/extensions/install >> ${dir}/packages.txt
echo k8s.io/kubernetes/pkg/apis/imagepolicy/install >> ${dir}/packages.txt
echo k8s.io/kubernetes/pkg/apis/policy/install >> ${dir}/packages.txt
echo k8s.io/kubernetes/pkg/apis/rbac/install >> ${dir}/packages.txt
echo k8s.io/kubernetes/pkg/apis/storage/install >> ${dir}/packages.txt
go list -f {{.Deps}} k8s.io/kubernetes/pkg/api/install | sed -e 's/ /\n/g' - | grep k8s.io | grep -v vendor >> ${dir}/packages.txt
go list -f {{.Deps}} k8s.io/kubernetes/pkg/apis/apps/install | sed -e 's/ /\n/g' - | grep k8s.io | grep -v vendor >> ${dir}/packages.txt
go list -f {{.Deps}} k8s.io/kubernetes/pkg/apis/authentication/install | sed -e 's/ /\n/g' - | grep k8s.io | grep -v vendor >> ${dir}/packages.txt
go list -f {{.Deps}} k8s.io/kubernetes/pkg/apis/authorization/install | sed -e 's/ /\n/g' - | grep k8s.io | grep -v vendor >> ${dir}/packages.txt
go list -f {{.Deps}} k8s.io/kubernetes/pkg/apis/autoscaling/install | sed -e 's/ /\n/g' - | grep k8s.io | grep -v vendor >> ${dir}/packages.txt
go list -f {{.Deps}} k8s.io/kubernetes/pkg/apis/batch/install | sed -e 's/ /\n/g' - | grep k8s.io | grep -v vendor >> ${dir}/packages.txt
go list -f {{.Deps}} k8s.io/kubernetes/pkg/apis/certificates/install | sed -e 's/ /\n/g' - | grep k8s.io | grep -v vendor >> ${dir}/packages.txt
go list -f {{.Deps}} k8s.io/kubernetes/pkg/apis/extensions/install | sed -e 's/ /\n/g' - | grep k8s.io | grep -v vendor >> ${dir}/packages.txt
go list -f {{.Deps}} k8s.io/kubernetes/pkg/apis/imagepolicy/install | sed -e 's/ /\n/g' - | grep k8s.io | grep -v vendor >> ${dir}/packages.txt
go list -f {{.Deps}} k8s.io/kubernetes/pkg/apis/policy/install | sed -e 's/ /\n/g' - | grep k8s.io | grep -v vendor >> ${dir}/packages.txt
go list -f {{.Deps}} k8s.io/kubernetes/pkg/apis/rbac/install | sed -e 's/ /\n/g' - | grep k8s.io | grep -v vendor >> ${dir}/packages.txt
go list -f {{.Deps}} k8s.io/kubernetes/pkg/apis/storage/install | sed -e 's/ /\n/g' - | grep k8s.io | grep -v vendor >> ${dir}/packages.txt
# used by tests
# echo k8s.io/kubernetes/pkg/util/diff >> ${dir}/packages.txt
# go list -f {{.Deps}} k8s.io/kubernetes/pkg/util/diff | sed -e 's/ /\n/g' - | grep k8s.io | grep -v vendor >> ${dir}/packages.txt
LC_ALL=C sort -u -o ${dir}/packages.txt ${dir}/packages.txt

echo "moving these packages"
cat ${dir}/packages.txt

# copy all the packages over
while read package; do
	unprefix_package=$(echo ${package} | sed 's|k8s.io/kubernetes/||g')
	mkdir -p ${KUBE_ROOT}/staging/src/k8s.io/apis/${unprefix_package}
	cp ${KUBE_ROOT}/${unprefix_package}/* ${KUBE_ROOT}/staging/src/k8s.io/apis/${unprefix_package} || true
	find ${KUBE_ROOT}/${unprefix_package} -maxdepth 1 -type f | xargs rm
done <${dir}/packages.txt

# need to remove the bazel files or bazel fails when this moves into vendor
find ${KUBE_ROOT}/staging/src/k8s.io/apis -name BUILD | xargs rm

# need to rewrite all the package imports for k8s.io/kuberentes to k8s.io/apis
find ${KUBE_ROOT}/staging/src/k8s.io/apis -name "*.go" | xargs sed -i 's|k8s.io/kubernetes|k8s.io/apis|g'

# need to rewrite all the package imports for these packages in the main repo to use the vendored copy
while read package; do
	echo "rewriting import for ${package}"
	new_package=$(echo ${package} | sed 's|k8s.io/kubernetes|k8s.io/apis|g')
	find ${KUBE_ROOT}/cmd ${KUBE_ROOT}/examples ${KUBE_ROOT}/federation ${KUBE_ROOT}/pkg ${KUBE_ROOT}/plugin ${KUBE_ROOT}/test -name "*.go" | xargs sed -i "s|${package}\"|${new_package}\"|g"
done <${dir}/packages.txt

# we don't want to rewrite imports for the packages we're modifying.  So check those back out, but only the files directly in that directory, not subdirs
# also, add .readonly files to each folder we moved
while read package; do
	unprefix_package=$(echo ${package} | sed 's|k8s.io/kubernetes/||g')
	find ${unprefix_package} -type f -maxdepth 1 | xargs git checkout
	touch ${unprefix_package}/.readonly
done <${dir}/packages.txt

# this file generates something or other, but we don't want to accidentally have it generate into an apis package
git checkout cmd/libs/go2idl/set-gen/main.go


# now run gofmt to get the sorting right
echo "running gofmt"
gofmt -s -w ${KUBE_ROOT}/cmd ${KUBE_ROOT}/examples ${KUBE_ROOT}/federation ${KUBE_ROOT}/pkg ${KUBE_ROOT}/plugin ${KUBE_ROOT}/test 

echo "run bazel"
${KUBE_ROOT}/hack/update-bazel.sh

