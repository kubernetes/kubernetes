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

# STEP II. copy utility functions in pkg/api/..., 
# TODO: they probably should live in pkg/util/
cp "${KUBE_ROOT}"/pkg/api/resource_helpers.go "${KUBE_ROOT}"/pkg/api/v1/resource_helpers.go
sed -i "s|package api|package v1|g" "${KUBE_ROOT}"/pkg/api/v1/resource_helpers.go

cp "${KUBE_ROOT}"/pkg/api/resource_helpers_test.go "${KUBE_ROOT}"/pkg/api/v1/resource_helpers_test.go
sed -i "s|package api|package v1|g" "${KUBE_ROOT}"/pkg/api/v1/resource_helpers_test.go
#================
helpers_file="${KUBE_ROOT}"/pkg/api/v1/helpers.go
readonly helpers_file
cp "${KUBE_ROOT}"/pkg/api/helpers.go "${helpers_file}"
sed -i "s|package api|package v1|g" "${helpers_file}"
echo "
type Sysctl struct {
	Name string
	Value string
}" >> "${helpers_file}"
sed -i "s|FieldSelector:   fields\(.*\),$|FieldSelector:   fields\1.String(),|g" "${helpers_file}"
sed -i "s|ResourceOpaqueIntPrefix|api.ResourceOpaqueIntPrefix|g" "${helpers_file}"
sed -i "s|\"k8s.io/kubernetes/pkg/api/unversioned\"|\"k8s.io/kubernetes/pkg/api/unversioned\"\n    \"k8s.io/kubernetes/pkg/api\"|g" "${helpers_file}"

# TODO: find a better place for this? NodeResources is originally defined in pkg/api. registry uses this type, so we need to keep the copy in pkg/api
echo "
// NodeResources is an object for conveying resource information about a node.
// see http://releases.k8s.io/HEAD/docs/design/resources.md for more details.
type NodeResources struct {
	// Capacity represents the available resources of a node
	Capacity ResourceList
}" >> "${helpers_file}"

cp "${KUBE_ROOT}"/pkg/api/helpers_test.go "${KUBE_ROOT}"/pkg/api/v1/helpers_test.go
sed -i "s|package api|package v1|g" "${KUBE_ROOT}"/pkg/api/v1/helpers_test.go
#================
#NOTE: these packages are used by pkg/validation as well, so need to duplicate them.
cp -r ${KUBE_ROOT}/pkg/api/pod ${KUBE_ROOT}/pkg/api/v1/pod
cp -r ${KUBE_ROOT}/pkg/api/service ${KUBE_ROOT}/pkg/api/v1/service
cp -r ${KUBE_ROOT}/pkg/api/endpoints ${KUBE_ROOT}/pkg/api/v1/endpoints
cp -r ${KUBE_ROOT}/pkg/apis/storage/util ${KUBE_ROOT}/pkg/apis/storage/v1beta1/util
#================


#================
# Build a list of files that need to be converted
files_to_convert=$(mktemp -p "${KUBE_ROOT}" files_to_convert.XXX)

cleanup() {
    rm -rf "${files_to_convert}"
}
trap cleanup EXIT SIGINT

cd "${KUBE_ROOT}" > /dev/null
# pkg/proxy
# pkg/kubelet

#TODO:
# kubectl uses pkg/kubelet/qos
find ./ -type f -name "*.go" \
    \( \
        -path './pkg/api/v1/pod/*' -o \
        -path './pkg/api/v1/service/*' -o \
        -path './pkg/apis/storage/v1beta1/util/*' -o \
        -path './pkg/api/v1/endpoints/*' -o \
        -path './pkg/controller/*' -o \
        -path './pkg/serviceaccount/*' -o \
        -path './pkg/fieldpath/*' -o \
        -path './pkg/cloudprovider/*' -o \
        -path './pkg/util/pod/*' -o \
        -path './pkg/util/replicaset/*' -o \
        -path './pkg/util/node/*' -o \
        -path './pkg/util/system/*' -o \
        -path './pkg/volume/*' -o \
        -path './pkg/kubelet/qos/*' -o \
        -path './plugin/pkg/scheduler/*' -o \
        -path './pkg/quota/*' \
    \) -print0 > "${files_to_convert}"

cat "${files_to_convert}" | while read -r -d $'\0' target; do
#target="$(cat ${files_to_convert})"
#readonly target

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

sed -i "s|CoreInterface|CoreV1Interface|g" "${target}"

sed -i "s|unversionedcore|v1core|g" "${target}"

sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/extensions/internalversion|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/extensions/v1beta1|g" "${target}"

sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/policy/internalversion|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/policy/v1alpha1|g" "${target}"

sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/apps/internalversion|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/apps/v1alpha1|g" "${target}"

sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/autoscaling/internalversion|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/autoscaling/v1|g" "${target}"

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
apps \"k8s.io/kubernetes/pkg/apis/apps/v1alpha1\"\n appsinternal \"k8s.io/kubernetes/pkg/apis/apps\"|g" "${target}"

sed -i "s|\"k8s.io/kubernetes/pkg/apis/policy\"|\
policy \"k8s.io/kubernetes/pkg/apis/policy/v1alpha1\"\n policyinternal \"k8s.io/kubernetes/pkg/apis/policy\"|g" "${target}"

sed -i "s|\"k8s.io/kubernetes/pkg/apis/rbac\"|\
rbac \"k8s.io/kubernetes/pkg/apis/rbac/v1alpha1\"\n rbacinternal \"k8s.io/kubernetes/pkg/apis/rbac\"|g" "${target}"

sed -i "s|\"k8s.io/kubernetes/pkg/apis/certificates\"|\
certificates \"k8s.io/kubernetes/pkg/apis/certificates/v1alpha1\"\n certificatesinternal \"k8s.io/kubernetes/pkg/apis/certificates\"|g" "${target}"

# needs to treat batch carefully, it has two versions
if [[ "${target}" == *pkg/controller/job* ]]; then
sed -i "s|\"k8s.io/kubernetes/pkg/apis/batch\"|\
batch \"k8s.io/kubernetes/pkg/apis/batch/v1\"\n batchinternal \"k8s.io/kubernetes/pkg/apis/batch\"|g" "${target}"
fi

if [[ "${target}" == *pkg/controller/scheduledjob* ]]; then
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
sed -i 's|v1.Scheme|api.Scheme|g' "${target}"
sed -i 's|v1.Unversioned|api.Unversioned|g' "${target}"
sed -i 's|v1.StrategicMergePatchType|api.StrategicMergePatchType|g' "${target}"
sed -i 's|v1.ObjectNameField|api.ObjectNameField|g' "${target}"
sed -i 's|v1.SecretTypeField|api.SecretTypeField|g' "${target}"
sed -i 's|v1.PodHostField|api.PodHostField|g' "${target}"
sed -i 's|v1.SecretTypeField|api.SecretTypeField|g' "${target}"
sed -i 's|v1.Codecs|api.Codecs|g' "${target}"
sed -i 's|v1.PatchType|api.PatchType|g' "${target}"
sed -i 's|\<v1.WithNamespace|api.WithNamespace|g' "${target}"
sed -i 's|\<v1.NewContext|api.NewContext|g' "${target}"
sed -i 's|\<v1.Kind(|api.Kind(|g' "${target}"

sed -i "s|\<v1\.Resource(|api.Resource(|g" "${target}"
sed -i "s|\<rbac\.Resource(|rbacinternal.Resource(|g" "${target}"
sed -i "s|\<extensions\.Resource(|extensionsinternal.Resource(|g" "${target}"
# Don't rewrite metrics_api to metrics_v1
sed -i "s|metrics_v1|metrics_api|g" "${target}"

#PART VI: rewrite labelselectors, to call .String()
#http://stackoverflow.com/questions/9053100/sed-regex-and-substring-negation
sed -i "/unversioned.LabelSelector/b; s/\<LabelSelector\(.*\)}/LabelSelector\1.String()}/g" "${target}"
sed -i "s/FieldSelector =\(.*\)$/FieldSelector =\1.String()/g" "${target}"

# PART VII: corner cases
# scalestatus.selector is map[string]string in v1beta1 and unversioned.Selector in extensions...
if [[ "${target}" == *pkg/controller/podautoscaler* ]]; then
sed -i "s,\
unversioned.LabelSelectorAsSelector(scale.Status.Selector),\
unversioned.LabelSelectorAsSelector(\&unversioned.LabelSelector{MatchLabels: scale.Status.Selector}),g" "${target}"
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
    fi
fi

#gofmt -w "${target}"
#goimports -w "${target}"
echo "processed ${target}"

done

cat ${files_to_convert} | xargs -0 goimports -w

time1=`date +%s`

echo "total runtime $((time1-time0))"

# Special cases:
#===================
storage="${KUBE_ROOT}/pkg/storage"
readonly storage
find "${storage}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,api\.ListOptions,v1\.ListOptions,g"
find "${storage}" -type f -name "*.go" -print0 | xargs -0 sed -i "s|\
k8s.io/kubernetes/pkg/api\"|\
k8s.io/kubernetes/pkg/api\"\n\"k8s.io/kubernetes/pkg/api/v1\"|g"
goimports -w "${storage}"
#===================
volume="${KUBE_ROOT}/pkg/volume"
find "${volume}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,FieldSelector: podSelector,FieldSelector: podSelector.String(),g"
find "${volume}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,FieldSelector: eventSelector,FieldSelector: eventSelector.String(),g"
goimports -w "${volume}"
#===================
deployment_util="${KUBE_ROOT}/pkg/controller/deployment/util"
find "${deployment_util}" -type f -name "*.go" -print0 | xargs -0 sed -i "s|\
ResolveFenceposts(\&deployment.Spec.Strategy.RollingUpdate.MaxSurge, \&deployment.Spec.Strategy.RollingUpdate.MaxUnavailable|\
ResolveFenceposts(deployment.Spec.Strategy.RollingUpdate.MaxSurge, deployment.Spec.Strategy.RollingUpdate.MaxUnavailable|g"

find "${deployment_util}" -type f -name "*.go" -print0 | xargs -0 sed -i "s|\
GetValueFromIntOrPercent(\&deployment.Spec.Strategy.RollingUpdate.MaxSurge|\
GetValueFromIntOrPercent(deployment.Spec.Strategy.RollingUpdate.MaxSurge|g"

#===================
certificates_controller="${KUBE_ROOT}/pkg/controller/certificates/controller.go"
find "${certificates_controller}" -type f -name *.go -print0 | xargs -0 sed -i "s|\
ParseCSR|ParseCSRV1alpha1|g"
#===================
#TODO: This pattern might be useful elsewhere, too
controllers="${KUBE_ROOT}/pkg/controller"
find "${controllers}" -type f -name *.go -print0 | xargs -0 sed -i "s|\
int32(replicas)|\
func() *int32 { i := int32(replicas); return \&i }()|g"
#===================
#TODO: might be useful elsewhere
node_util="${KUBE_ROOT}/pkg/controller/node/"
find "${node_util}" -type f -name *controller_utils.go -print0 | xargs -0 sed -i "s|\
fields.OneTermEqualSelector(api.PodHostField, nodeName)|\
fields.OneTermEqualSelector(api.PodHostField, nodeName).String()|g"
find "${node_util}" -type f -name *nodecontroller.go -print0 | xargs -0 sed -i "s|\
Everything()|\
Everything().String()|g"
#===================
#TODO: might be useful elsewhere
deployment_sync="${KUBE_ROOT}/pkg/controller/deployment"
find "${deployment_sync}" -type f -name *sync.go -print0 | xargs -0 sed -i "s|\
\"k8s.io/kubernetes/pkg/api\"|\
\"k8s.io/kubernetes/pkg/api\"\n\
\"k8s.io/kubernetes/pkg/labels\"|g"

find "${deployment_sync}" -type f -name *sync.go -print0 | xargs -0 sed -i "s|\
^\(.*\)List(options.LabelSelector)|\
parsed, err := labels.Parse(options.LabelSelector)\n\
if err != nil {\n\
    return nil, err\n\
}\n\
\1List(parsed)|g"

find "${deployment_sync}" -type f -name *sync.go -print0 | xargs -0 sed -i "s|\
Replicas:        0|\
Replicas: func(i int32) *int32 { return \&i }(0)|g"
#===================
# TODO: First 2 patterns occurred twice!
quota="${KUBE_ROOT}/pkg/quota"
find "${quota}" -type f -name *evaluator.go -print0 | xargs -0 sed -i "s|\
^\(.*\)List(options.LabelSelector)|\
parsed, err := labels.Parse(options.LabelSelector)\n\
if err != nil {\n\
    return nil, err\n\
}\n\
\1List(parsed)|g"

find "${quota}" -type f -name *evaluator.go -print0 | xargs -0 sed -i "s|\
Everything()|\
Everything().String()|g"

find "${quota}" -type f -name *pods.go -print0 | xargs -0 sed -i "s|\
\"k8s.io/kubernetes/pkg/api/validation\"|\
\"k8s.io/kubernetes/pkg/api/v1/validation\"|g"

#===================
tokengetter="${KUBE_ROOT}/pkg/controller/serviceaccount"
find "${tokengetter}" -type f -name *tokengetter.go -print0 | xargs -0 sed -i "s|\
return r.serviceAccounts.GetServiceAccount(ctx, name)|\
internalServiceAccount, err := r.serviceAccounts.GetServiceAccount(ctx, name)\n\
	if err != nil {\n\
		return nil, err\n\
	}\n\
	v1ServiceAccount := v1.ServiceAccount{}\n\
	err = v1.Convert_api_ServiceAccount_To_v1_ServiceAccount(internalServiceAccount, \&v1ServiceAccount, nil)\n\
	return \&v1ServiceAccount, err\n|g"

find "${tokengetter}" -type f -name *tokengetter.go -print0 | xargs -0 sed -i "s|\
return r.secrets.GetSecret(ctx, name)|\
internalSecret, err := r.secrets.GetSecret(ctx, name)\n\
	if err != nil {\n\
		return nil, err\n\
	}\n\
	v1Secret := v1.Secret{}\n\
	err = v1.Convert_api_Secret_To_v1_Secret(internalSecret, \&v1Secret, nil)\n\
	return \&v1Secret, err\n|g"
#===================
scheduledjob="${KUBE_ROOT}/pkg/controller/scheduledjob"
find "${scheduledjob}" -type f -name *.go -print0 | xargs -0 sed -i "s|\
Batch()\.|\
BatchV2alpha1().|g"

# TODO: need to discuss how to solve this kind of duplicate. Type alias?
echo "
func IsJobFinished(j *batch.Job) bool {
	for _, c := range j.Status.Conditions {
		if (c.Type == batch.JobComplete || c.Type == batch.JobFailed) && c.Status == v1.ConditionTrue {
			return true
		}
	}
	return false
}" >> "${scheduledjob}"/utils.go

find "${scheduledjob}" -type f -name *.go -print0 | xargs -0 sed -i "s|\
job.IsJobFinished(|\
IsJobFinished(|g"
goimports -w "${scheduledjob}"
#===================


