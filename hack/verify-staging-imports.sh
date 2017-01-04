#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env


for dep in $(find staging/src/k8s.io/* -maxdepth 0 -type d); do
	set +e
	dep=$(basename ${dep})
	go list -f {{.Deps}} ./vendor/k8s.io/${dep}/... | sed -e 's/ /\n/g' - | grep k8s.io/kubernetes | grep -v vendor | LC_ALL=C sort -u 
	hasKubeDep=$?
	if [ "${hasKubeDep}" -eq "0" ]; then
		echo "${dep} has a cyclical dependency"
		exit 1
	fi
done

exit 0