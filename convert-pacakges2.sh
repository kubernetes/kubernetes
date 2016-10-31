#!/bin/bash
# convert to versioned API
set -o errexit
set -o nounset
set -o pipefail

time0=`date +%s`

KUBE_ROOT=$(dirname "${BASH_SOURCE}")
source "${KUBE_ROOT}/hack/lib/init.sh"

# STEP I. convert pkg/client first
"${KUBE_ROOT}"/convert-client.sh

# STEP II. copy utility functions in pkg/api/...

# Build a list of files that need to be converted
files_to_convert=$(mktemp -p "${KUBE_ROOT}" files_to_convert.XXX)

cleanup() {
    rm -rf "${files_to_convert}"
}
trap cleanup EXIT SIGINT

cd "${KUBE_ROOT}" > /dev/null
find ./ -type f -name "*.go" \
    \( \
        -path './pkg/controller/*' -o \
        -path './pkg/serviceaccount/*' -o \
        -path './pkg/fieldpath/*' -o \
        -path './pkg/volume/*' \
    \) -print0 > "${files_to_convert}"

#cat "${files_to_convert}" | while read -r -d $'\0' target; do
target="$(cat ${files_to_convert})"
readonly target

# PART I: convert client imports
cat ${files_to_convert} | xargs -0 sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5\
|g"

cat ${files_to_convert} | xargs -0 sed -i "s|\
^\"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5\"|\
clientset \"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5\"\
|g"

cat ${files_to_convert} | xargs -0 sed -i "s|\
internalclientset\.|\
clientset.\
|g"

# PART I.1: corner cases
cat ${files_to_convert} | xargs -0 sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/core/internalversion|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/core/v1|g"

cat ${files_to_convert} | xargs -0 sed -i "s|unversionedcore|v1core|g"

cat ${files_to_convert} | xargs -0 sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/extensions/internalversion|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/extensions/v1beta1|g"

cat ${files_to_convert} | xargs -0 sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/policy/internalversion|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/policy/v1alpha1|g"

cat ${files_to_convert} | xargs -0 sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/apps/internalversion|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/apps/v1alpha1|g"

cat ${files_to_convert} | xargs -0 sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/autoscaling/internalversion|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/autoscaling/v1|g"

# PART II: convert type imports
cat ${files_to_convert} | xargs -0 sed -i "s|\
k8s.io/kubernetes/pkg/api\"|\
k8s.io/kubernetes/pkg/api\"\n\"k8s.io/kubernetes/pkg/api/v1\"|g"
# change apis from unversioned to versioned
cat ${files_to_convert} | xargs -0 sed -i "s|\"k8s.io/kubernetes/pkg/apis/storage\"|storage \"k8s.io/kubernetes/pkg/apis/storage/v1beta1\"|g"
cat ${files_to_convert} | xargs -0 sed -i "s|\"k8s.io/kubernetes/pkg/apis/extensions\"|extensions \"k8s.io/kubernetes/pkg/apis/extensions/v1beta1\"|g"
# special case
cat ${files_to_convert} | grep -zZ -v "replica_set.go" | xargs -0 sed -i "s,v1beta1.SchemeGroupVersion,extensions.SchemeGroupVersion,g"

cat ${files_to_convert} | xargs -0 sed -i "s|\"k8s.io/kubernetes/pkg/apis/autoscaling\"|autoscaling \"k8s.io/kubernetes/pkg/apis/autoscaling/v1\"|g"
cat ${files_to_convert} | xargs -0 sed -i "s|\"k8s.io/kubernetes/pkg/apis/apps\"|apps \"k8s.io/kubernetes/pkg/apis/apps/v1alpha1\"|g"
cat ${files_to_convert} | xargs -0 sed -i "s|\"k8s.io/kubernetes/pkg/apis/policy\"|policy \"k8s.io/kubernetes/pkg/apis/policy/v1alpha1\"|g"
# needs to treat batch carefully, it has two versions
cat ${files_to_convert} | grep -zZ "pkg/controller/job" | xargs -0 sed -i "s|\"k8s.io/kubernetes/pkg/apis/batch\"|batch \"k8s.io/kubernetes/pkg/apis/batch/v1\"|g"

cat ${files_to_convert} | grep -zZ "pkg/controller/scheduledjob" | xargs -0 sed -i "s|\"k8s.io/kubernetes/pkg/apis/batch\"|batch \"k8s.io/kubernetes/pkg/apis/batch/v2alpha1\"|g"

# PART III: rewrite api. to v1.
#cat ${files_to_convert} | xargs -0 sed -i "s|api\.|v1.|g"
cat ${files_to_convert} | xargs -0 sed -i 's|api\.|v1\.|g'
cat ${files_to_convert} | xargs -0 sed -i 's|v1.Scheme|api.Scheme|g'
cat ${files_to_convert} | xargs -0 sed -i 's|v1.Unversioned|api.Unversioned|g'
cat ${files_to_convert} | xargs -0 sed -i 's|v1.StrategicMergePatchType|api.StrategicMergePatchType|g'
cat ${files_to_convert} | xargs -0 sed -i 's|v1.ObjectNameField|api.ObjectNameField|g'
cat ${files_to_convert} | xargs -0 sed -i 's|v1.SecretTypeField|api.SecretTypeField|g'
cat ${files_to_convert} | xargs -0 sed -i 's|v1.PodHostField|api.PodHostField|g'
cat ${files_to_convert} | xargs -0 sed -i 's|v1.SecretTypeField|api.SecretTypeField|g'
cat ${files_to_convert} | xargs -0 sed -i 's|v1.Codecs|api.Codecs|g'
cat ${files_to_convert} | xargs -0 sed -i 's|v1.PatchType|api.PatchType|g'
cat ${files_to_convert} | xargs -0 sed -i "s|v1\.Resource(|api\.Resource(|g"

# Don't rewrite metrics_api to metrics_v1
cat ${files_to_convert} | xargs -0 sed -i "s|metrics_v1|metrics_api|g"

#PART VI: rewrite labelselectors, to call .String()
#http://stackoverflow.com/questions/9053100/cat ${files_to_convert} | xargs -0 sed-regex-and-substring-negation
cat ${files_to_convert} | xargs -0 sed -i "/unversioned.LabelSelector/b; s/\<LabelSelector\(.*\)}/LabelSelector\1.String()}/g"
cat ${files_to_convert} | xargs -0 sed -i "s/FieldSelector =\(.*\)$/FieldSelector =\1.String()/g"

# PART VII: corner cases
# scalestatus.selector is map[string]string in v1beta1 and unversioned.Selector in extensions...
cat ${files_to_convert} | grep -zZ pkg/controller/podautoscaler | xargs -0 sed -i "s,\
unversioned.LabelSelectorAsSelector(scale.Status.Selector),\
unversioned.LabelSelectorAsSelector(\&unversioned.LabelSelector{MatchLabels: scale.Status.Selector}),g"

# *int32 to int32
cat ${files_to_convert} | grep -zZ pkg/controller | grep -zZ -v horizontal | xargs -0 sed -i "s,\([a-zA-Z0-9]\+\)\.Spec.Replicas,*(\1.Spec.Replicas),g"
cat ${files_to_convert} | grep -zZ pkg/controller/deployment/sync_test.go | xargs -0 sed -i "s|test\.\*(|*(test.|g" 
cat ${files_to_convert} | grep -zZ pkg/controller/petset/iterator.go | xargs -0 sed -i "s|pi.\*(|*(pi.|g" 
cat ${files_to_convert} | grep -zZ pkg/controller/controller_utils.go | xargs -0 sed -i "s,o\[i\].Spec.Replicas > o\[j\].Spec.Replicas,*(o[i].Spec.Replicas) > *(o[j].Spec.Replicas),g"

#gofmt -w
#goimports -w
echo "procescat ${files_to_convert} | xargs -0 sed ${target}"

cat ${files_to_convert} | xargs -0 goimports -w

time1=`date +%s`

echo "total runtime $((time1-time0))"

