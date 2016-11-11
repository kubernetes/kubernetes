#!/bin/bash
# convert to versioned API
set -o errexit
set -o nounset
set -o pipefail

time0=`date +%s`

KUBE_ROOT=$(dirname "${BASH_SOURCE}")
source "${KUBE_ROOT}/hack/lib/init.sh"

#================
# Build a list of files that need to be converted
files_to_convert=$(mktemp -p "${KUBE_ROOT}" files_to_convert.XXX)

cleanup() {
    rm -rf "${files_to_convert}"
}
trap cleanup EXIT SIGINT

cd "${KUBE_ROOT}" > /dev/null

#TODO:

find ./ -type f -name "*.go" \
    \( \
        -path './test/e2e_node/*' \
    \) -print0 > "${files_to_convert}"

cat "${files_to_convert}" | while read -r -d $'\0' target; do

# PART I: convert client imports
sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5\
|g" "${target}"

sed -i "s|\
^\(\s\)*\"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5\"|\
\1clientset \"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5\"\
|g" "${target}"

sed -i "s|\
internalclientset\.|\
clientset.\
|g" "${target}"

# PART I.1: corner cases
sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/core/internalversion|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/core/v1|g" "${target}"

# name the import
#sed -i "s|\
#^\(\s\)*\"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/core/v1\"|\
#\1v1core \"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/core/v1\"\
#|g" "${target}"

#sed -i "s|internalversion\.|v1core.|g" "${target}"

sed -i "s|CoreInterface|CoreV1Interface|g" "${target}"

sed -i "s|unversionedcore|v1core|g" "${target}"

sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/extensions/internalversion|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/extensions/v1beta1|g" "${target}"

sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/policy/internalversion|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/policy/v1alpha1|g" "${target}"

# TODO: this will change after rebase
if [[ "${target}" == *pet_set_test.go ]]; then
sed -i "s|\
internalversion.AppsInterface|\
v1beta1.AppsV1beta1Interface|g" "${target}"

sed -i "s|\
internalversion.StatefulSetInterface|\
v1beta1.StatefulSetInterface|g" "${target}"

sed -i "s|\
fake.FakeApps|\
fake.FakeAppsV1beta1|g" "${target}"
fi

sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/apps/internalversion|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/apps/v1beta1|g" "${target}"

sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/autoscaling/internalversion|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/autoscaling/v1|g" "${target}"

sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/certificates/internalversion|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/certificates/v1alpha1|g" "${target}"

sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/rbac/internalversion|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/rbac/v1alpha1|g" "${target}"

# PART II: convert type imports
sed -i "s|\
k8s.io/kubernetes/pkg/api\"|\
k8s.io/kubernetes/pkg/api\"\n\"k8s.io/kubernetes/pkg/api/v1\"|g" "${target}"
# change apis from unversioned to versioned
sed -i "s|\"k8s.io/kubernetes/pkg/apis/storage\"|\
storage \"k8s.io/kubernetes/pkg/apis/storage/v1beta1\"\n    storageinternal \"k8s.io/kubernetes/pkg/apis/storage\"|g" "${target}"

sed -i "s|\"k8s.io/kubernetes/pkg/apis/extensions\"|\
extensions \"k8s.io/kubernetes/pkg/apis/extensions/v1beta1\"\n  extensionsinternal \"k8s.io/kubernetes/pkg/apis/extensions\"|g" "${target}"

# special case
if [[ "${target}" != *replica_set.go ]]; then
    sed -i "s,v1beta1.SchemeGroupVersion,extensions.SchemeGroupVersion,g" "${target}"
fi

sed -i "s|\"k8s.io/kubernetes/pkg/apis/autoscaling\"|\
autoscaling \"k8s.io/kubernetes/pkg/apis/autoscaling/v1\"\n autoscalinginternal \"k8s.io/kubernetes/pkg/apis/autoscaling\"|g" "${target}"

sed -i "s|\"k8s.io/kubernetes/pkg/apis/apps\"|\
apps \"k8s.io/kubernetes/pkg/apis/apps/v1beta1\"\n appsinternal \"k8s.io/kubernetes/pkg/apis/apps\"|g" "${target}"

sed -i "s|\"k8s.io/kubernetes/pkg/apis/policy\"|\
policy \"k8s.io/kubernetes/pkg/apis/policy/v1alpha1\"\n policyinternal \"k8s.io/kubernetes/pkg/apis/policy\"|g" "${target}"

sed -i "s|\"k8s.io/kubernetes/pkg/apis/rbac\"|\
rbac \"k8s.io/kubernetes/pkg/apis/rbac/v1alpha1\"\n rbacinternal \"k8s.io/kubernetes/pkg/apis/rbac\"|g" "${target}"

sed -i "s|rbac rbac|rbac|g" "${target}"

sed -i "s|\"k8s.io/kubernetes/pkg/apis/certificates\"|\
certificates \"k8s.io/kubernetes/pkg/apis/certificates/v1alpha1\"\n certificatesinternal \"k8s.io/kubernetes/pkg/apis/certificates\"|g" "${target}"

# needs to treat batch carefully, it has two versions
if [[ "${target}" == *v1_jobs* ]] || [[ "${target}" == */job.go* ]]; then
sed -i "s|\"k8s.io/kubernetes/pkg/apis/batch\"|\
batch \"k8s.io/kubernetes/pkg/apis/batch/v1\"\n batchinternal \"k8s.io/kubernetes/pkg/apis/batch\"|g" "${target}"
fi

if [[ "${target}" == *cronjob* ]]; then
sed -i "s|\"k8s.io/kubernetes/pkg/apis/batch\"|\
batch \"k8s.io/kubernetes/pkg/apis/batch/v2alpha1\"\n   batchinternal \"k8s.io/kubernetes/pkg/apis/batch\"|g" "${target}"
fi

#PART III: utility functions rewrites
sed -i "s|\"k8s.io/kubernetes/pkg/api/pod\"|\
\"k8s.io/kubernetes/pkg/api/v1/pod\"|g" "${target}"

sed -i "s|\"k8s.io/kubernetes/pkg/api/service\"|\
\"k8s.io/kubernetes/pkg/api/v1/service\"|g" "${target}"

sed -i "s|\"k8s.io/kubernetes/pkg/api/endpoints\"|\
\"k8s.io/kubernetes/pkg/api/v1/endpoints\"|g" "${target}"

sed -i "s|\"k8s.io/kubernetes/pkg/apis/storage/util\"|\
\"k8s.io/kubernetes/pkg/apis/storage/v1beta1/util\"|g" "${target}"

# PART III: rewrite api. to v1.
#sed -i "s|api\.|v1.|g" "${target}"
sed -i 's|\<api\.|v1\.|g' "${target}"
sed -i 's|v1.Scheme\>|api.Scheme|g' "${target}"
sed -i 's|v1.Unversioned|api.Unversioned|g' "${target}"
sed -i 's|v1.StrategicMergePatchType|api.StrategicMergePatchType|g' "${target}"
sed -i 's|v1.MergePatchType|api.MergePatchType|g' "${target}"
sed -i 's|v1.JSONPathType|api.JSONPatchType|g' "${target}"
sed -i 's|v1.ObjectNameField|api.ObjectNameField|g' "${target}"
sed -i 's|v1.SecretTypeField|api.SecretTypeField|g' "${target}"
sed -i 's|v1.PodHostField|api.PodHostField|g' "${target}"
sed -i 's|v1.SecretTypeField|api.SecretTypeField|g' "${target}"
sed -i 's|v1.Codecs|api.Codecs|g' "${target}"
sed -i 's|v1.ParameterCodec|api.ParameterCodec|g' "${target}"
sed -i 's|v1.PatchType|api.PatchType|g' "${target}"
sed -i 's|\<v1.WithNamespace|api.WithNamespace|g' "${target}"
sed -i 's|\<v1.NamespaceValue|api.NamespaceValue|g' "${target}"
sed -i 's|\<v1.NewContext|api.NewContext|g' "${target}"
sed -i 's|\<v1.Context|api.Context|g' "${target}"
sed -i 's|\<v1.Kind(|api.Kind(|g' "${target}"
sed -i 's|\<extensions.Kind(|extensionsinternal.Kind(|g' "${target}"
sed -i 's|\<batch.Kind(|batchinternal.Kind(|g' "${target}"
sed -i 's|\<v1.ListMetaFor|api.ListMetaFor|g' "${target}"
sed -i 's|\<v1.StreamType|api.StreamType|g' "${target}"
sed -i 's|\<v1.PortHeader|api.PortHeader|g' "${target}"
sed -i 's|\<v1.PortForwardRequestIDHeader|api.PortForwardRequestIDHeader|g' "${target}"
sed -i 's|\<v1.Exec\([a-zA-Z]*\)Param|api.Exec\1Param|g' "${target}"
sed -i 's|\<v1.NamespaceSystem|api.NamespaceSystem|g' "${target}"

sed -i "s|\
reaper.Stop(\(.*\)v1.NewDeleteOptions(0))|\
reaper.Stop(\1api.NewDeleteOptions(0))|g" "${target}"

sed -i "s|\<v1\.Resource(|api.Resource(|g" "${target}"
sed -i "s|\<rbac\.Resource(|rbacinternal.Resource(|g" "${target}"
sed -i "s|\<extensions\.Resource(|extensionsinternal.Resource(|g" "${target}"
# Don't rewrite metrics_api to metrics_v1
sed -i "s|metrics_v1|metrics_api|g" "${target}"

#PART VI: rewrite labelselectors, to call .String()
#http://stackoverflow.com/questions/9053100/sed-regex-and-substring-negation
sed -i "/unversioned.LabelSelector/b; s/\<LabelSelector\(.*\)}/LabelSelector\1.String()}/g" "${target}"
sed -i "s/\<FieldSelector =\(.*\)$/FieldSelector =\1.String()/g" "${target}"
sed -i "s/\<LabelSelector =\(.*\)$/LabelSelector =\1.String()/g" "${target}"

# PART VII: corner cases
# scalestatus.selector is map[string]string in v1beta1 and unversioned.Selector in extensions...
if [[ "${target}" == *pkg/controller/podautoscaler* ]]; then
sed -i "s,\
unversioned.LabelSelectorAsSelector(scale.Status.Selector),\
unversioned.LabelSelectorAsSelector(\&unversioned.LabelSelector{MatchLabels: scale.Status.Selector}),g" "${target}"

sed -i "s|\
Selector: selector,|\
Selector: selector.MatchLabels,|g" "${target}"
fi

# *int32 to int32
if [[ "${target}" == *pkg/controller* ]]; then 
    if [[ "${target}" != *horizontal* ]]; then
        sed -i "s,\([a-zA-Z0-9]\+\)\.Spec.Replicas,*(\1.Spec.Replicas),g" "${target}"
    fi
    if [[ "${target}" == *deployment/sync_test.go* ]]; then
        sed -i "s|test\.\*(|*(test.|g" "${target}" 
    fi
    if [[ "${target}" == *petset/iterator.go* ]]; then
        sed -i "s|pi.\*(|*(pi.|g" "${target}" 
    fi
    if [[ "${target}" == *controller_utils.go* ]]; then
        sed -i "s,o\[i\].Spec.Replicas > o\[j\].Spec.Replicas,*(o[i].Spec.Replicas) > *(o[j].Spec.Replicas),g" "${target}"
        sed -i "s,o\[i\].Spec.Replicas == o\[j\].Spec.Replicas,*(o[i].Spec.Replicas) == *(o[j].Spec.Replicas),g" "${target}"
    fi
fi

#gofmt -w "${target}"
#goimports -w "${target}"
echo "processed ${target}"

# Special for remaining packages
if [[ "${target}" == *pkg/dns/* ]]; then 
    sed -i "s,kapi\.,v1.,g" "${target}"
fi

done

cat ${files_to_convert} | xargs -0 goimports -w

time1=`date +%s`

echo "total runtime $((time1-time0))"

#=========================================special cases

find "${KUBE_ROOT}/test/e2e_node" -type f -name "resource_collector.go" -print0 | xargs -0 sed -i "\
/SecurityContext: \&v1\.PodSecurityContext{/{
N
N
s|.*\n.*\n.*|\
HostNetwork: true,\n\
SecurityContext: \&v1.PodSecurityContext{},|g
}"

find "${KUBE_ROOT}/test/e2e_node" -type f -name "density_test.go" -print0 | xargs -0 sed -i "s|\
podType\.String()|\
podType|g"
