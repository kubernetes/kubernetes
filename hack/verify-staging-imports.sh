#!/bin/bash

# Copyright 2017 The Kubernetes Authors.
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

function print_forbidden_imports () {
    set -o errexit # this was unset by ||
    local PACKAGE="$1"
    shift
    local RE=""
    local SEP=""
    for CLAUSE in "$@"; do
        RE+="${SEP}${CLAUSE}"
        SEP='\|'
    done
    local FORBIDDEN=$(
        go list -f $'{{with $package := .ImportPath}}{{range $.Imports}}{{$package}} imports {{.}}\n{{end}}{{end}}' ./vendor/k8s.io/${PACKAGE}/... |
        sed 's|^k8s.io/kubernetes/vendor/||;s| k8s.io/kubernetes/vendor/| |' |
        grep -v " k8s.io/${PACKAGE}" |
        grep " k8s.io/" |
        grep -v -e "imports \(${RE}\)"
    )
    if [ -n "${FORBIDDEN}" ]; then
        echo "${PACKAGE} has a forbidden dependency:"
        echo
        echo "${FORBIDDEN}" | sed 's/^/  /'
        echo
        return 1
    fi
    local TEST_FORBIDDEN=$(
        go list -f $'{{with $package := .ImportPath}}{{range $.TestImports}}{{$package}} imports {{.}}\n{{end}}{{end}}' ./vendor/k8s.io/${PACKAGE}/... |
        sed 's|^k8s.io/kubernetes/vendor/||;s| k8s.io/kubernetes/vendor/| |' |
        grep -v " k8s.io/${PACKAGE}" |
        grep " k8s.io/" |
        grep -v -e "imports \(${RE}\)"
    )
    if [ -n "${TEST_FORBIDDEN}" ]; then
        echo "${PACKAGE} has a forbidden dependency in test code:"
        echo
        echo "${TEST_FORBIDDEN}" | sed 's/^/  /'
        echo
        return 1
    fi
    return 0
}

RC=0
print_forbidden_imports apimachinery || RC=1
print_forbidden_imports client-go k8s.io/apimachinery || RC=1
print_forbidden_imports apiserver k8s.io/apimachinery k8s.io/client-go || RC=1
print_forbidden_imports kube-aggregator k8s.io/apimachinery k8s.io/client-go k8s.io/apiserver || RC=1
print_forbidden_imports sample-apiserver k8s.io/apimachinery k8s.io/client-go k8s.io/apiserver || RC=1
print_forbidden_imports apiextensions-apiserver k8s.io/apimachinery k8s.io/client-go k8s.io/apiserver || RC=1
if [ ${RC} != 0 ]; then
    exit ${RC}
fi

if grep -rq '// import "k8s.io/kubernetes/' 'staging/'; then
	echo 'file has "// import "k8s.io/kubernetes/"'
	exit 1
fi

for EXAMPLE in vendor/k8s.io/client-go/examples/{in-cluster-client-configuration,out-of-cluster-client-configuration,third-party-resources-deprecated} vendor/k8s.io/apiextensions-apiserver/examples ; do
	test -d "${EXAMPLE}" # make sure example is still there
	if go list -f '{{ join .Deps "\n" }}' "./${EXAMPLE}/..." | sort | uniq | grep -q k8s.io/client-go/plugin; then
		echo "${EXAMPLE} imports client-go plugins by default, but shouldn't."
		exit 1
	fi
done

exit 0
