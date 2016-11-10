
#!/bin/bash
# convert to versioned API
set -o errexit
set -o nounset
set -o pipefail

time0=`date +%s`

KUBE_ROOT=$(dirname "${BASH_SOURCE}")
source "${KUBE_ROOT}/hack/lib/init.sh"

files_to_convert=$(mktemp -p "${KUBE_ROOT}" files_to_convert.XXX)
cleanup() {
    rm -rf "${files_to_convert}"
}
trap cleanup EXIT SIGINT

fedeartion="${KUBE_ROOT}/federation"

find "${fedeartion}/pkg/federation-controller" -type f -name *.go -print0 > "${files_to_convert}"

cat "${files_to_convert}" | while read -r -d $'\0' target; do

echo "processing ${target}"

v1name="v1"
if grep 'api_v1 "k8s.io/kubernetes/pkg/api/v1"' "${target}" > /dev/null; then
    v1name="api_v1"
elif grep 'apiv1 "k8s.io/kubernetes/pkg/api/v1"' "${target}" > /dev/null; then
    v1name="apiv1"
fi

sed -i "\
/ListFunc: func(options api\.ListOptions) /{
N
N
/targetClient == nil/b
s|api\.ListOptions|${v1name}\.ListOptions|g
s|versionedOptions|options|g
s|\(.*\n\).*\n\(.*\)|\1\2|g
}" "${target}"

sed -i "\
/WatchFunc: func(options api\.ListOptions) (watch\.Interface, error)/{
N
N
/targetClient == nil/b
s|api\.ListOptions|${v1name}\.ListOptions|g
s|versionedOptions|options|g
s|\(.*\n\).*\n\(.*\)|\1\2|g
}" "${target}"

sed -i "s|api\.EventSource|${v1name}.EventSource|g" "${target}"

done

goimports -w "${fedeartion}/pkg/federation-controller"
