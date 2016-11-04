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
# pkg/proxy
# pkg/kubelet

#TODO:
# kubectl uses pkg/kubelet/qos

        #-path './pkg/kubelet/config/file.go' -o \
        #-path './pkg/kubelet/config/http.go' -o \

find ./ -type f -name "*.go" \
    \( \
        -path './pkg/kubelet/*' -o \
        -path './pkg/security/apparmor/*' -o \
        -path './pkg/credentialprovider/*' -o \
        -path './pkg/securitycontext/*' \
    \) \
    -not \( \
        -path './pkg/kubelet/config/common.go' \
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
v1alpha1.AppsV1alpha1Interface|g" "${target}"

sed -i "s|\
internalversion.StatefulSetInterface|\
v1alpha1.StatefulSetInterface|g" "${target}"

sed -i "s|\
fake.FakeApps|\
fake.FakeAppsV1alpha1|g" "${target}"
fi

sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/apps/internalversion|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/apps/v1alpha1|g" "${target}"

sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/autoscaling/internalversion|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/autoscaling/v1|g" "${target}"

sed -i "s|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/certificates/internalversion|\
k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/certificates/v1alpha1|g" "${target}"

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
sed -i 's|v1.Scheme\>|api.Scheme|g' "${target}"
sed -i 's|v1.Unversioned|api.Unversioned|g' "${target}"
sed -i 's|v1.StrategicMergePatchType|api.StrategicMergePatchType|g' "${target}"
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
sed -i 's|\<v1.ListMetaFor|api.ListMetaFor|g' "${target}"
sed -i 's|\<v1.StreamType|api.StreamType|g' "${target}"
sed -i 's|\<v1.PortHeader|api.PortHeader|g' "${target}"
sed -i 's|\<v1.PortForwardRequestIDHeader|api.PortForwardRequestIDHeader|g' "${target}"
sed -i 's|\<v1.Exec\([a-zA-Z]*\)Param|api.Exec\1Param|g' "${target}"


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

#PART X. kubelet specific
sed -i "s,\
pod.Spec.SecurityContext.HostIPC,\
pod.Spec.HostIPC,g" "${target}"

sed -i "s,\
pod.Spec.SecurityContext.HostNetwork,\
pod.Spec.HostNetwork,g" "${target}"

sed -i "s,\
pod.Spec.SecurityContext.HostPID,\
pod.Spec.HostPID,g" "${target}"


#gofmt -w "${target}"
#goimports -w "${target}"
echo "processed ${target}"

done

cat ${files_to_convert} | xargs -0 goimports -w

time1=`date +%s`

echo "total runtime $((time1-time0))"

# Special cases:
#===================
securitycontext="${KUBE_ROOT}/pkg/securitycontext"
find "${securitycontext}" -type f -name *provider_test.go -print0 | xargs -0 sed -i "s|\
apitesting\.DeepEqualSafePodSpec|\
apitesting.V1DeepEqualSafePodSpec|g"

#===================
csr_util="${KUBE_ROOT}/pkg/kubelet/util/csr"
find "${csr_util}" -type f -name *csr.go -print0 | xargs -0 sed -i "s|\
fields.OneTermEqualSelector(\(.*\))|\
fields.OneTermEqualSelector(\1).String()|g"
#===================
container="${KUBE_ROOT}/pkg/kubelet/container"
find "${container}" -type f -name *helpers.go -print0 | xargs -0 sed -i "s|\
return pod.Spec.SecurityContext != nil \&\& pod.Spec.HostNetwork|\
return pod.Spec.HostNetwork|g"

hostport="${KUBE_ROOT}/pkg/kubelet/network/hostport"
find "${hostport}" -type f -name *hostport.go -print0 | xargs -0 sed -i "s|\
r.Pod.Spec.SecurityContext != nil \&\& r.Pod.Spec.SecurityContext.HostNetwork|\
r.Pod.Spec.HostNetwork|g"

#===================
dockertools="${KUBE_ROOT}/pkg/kubelet/dockertools"
find "${dockertools}" -type f -name *docker_manager.go -print0 | xargs -0 sed -i "s|\
pod.Spec.SecurityContext != nil \&\& pod.Spec.HostPID|\
pod.Spec.HostPID|g"

find "${dockertools}" -type f -name *docker_manager.go -print0 | xargs -0 sed -i "s|\
pod.Spec.SecurityContext != nil \&\& pod.Spec.HostIPC|\
pod.Spec.HostIPC|g"


#===================
sysctl="${KUBE_ROOT}/pkg/kubelet/sysctl"
find "${sysctl}" -type f -name *whitelist.go -print0 | xargs -0 sed -i "s|\
pod.Spec.SecurityContext.HostNetwork|\
pod.Spec.HostNetwork|g"

find "${sysctl}" -type f -name *whitelist.go -print0 | xargs -0 sed -i "s|\
pod.Spec.SecurityContext.HostIPC|\
pod.Spec.HostIPC|g"

#==================
kubelet="${KUBE_ROOT}/pkg/kubelet"
find "${kubelet}" -type f -name *kubelet_pods.go -print0 | xargs -0 sed -i "s|\
k8s.io/kubernetes/pkg/api/validation|\
k8s.io/kubernetes/pkg/api/v1/validation|g"

find "${kubelet}" -type f -name *kubelet_pods.go -print0 | xargs -0 sed -i "s|\
return pod.Spec.SecurityContext != nil \&\& pod.Spec.HostNetwork|\
return pod.Spec.HostNetwork|g"

find "${kubelet}" -type f -name *kubelet_node_status.go -print0 | xargs -0 sed -i "s|\
AsSelector()|\
AsSelector().String()|g"

find "${kubelet}" -type f \
    \(\
        -name *rkt.go -o \
        -name *rkt_test.go -o \
        -name *kubelet_pods_test.go -o \
        -name *kubelet_test.go -o \
        -name *docker_manager_test.go \
    \) -print0 | xargs -0 sed -i "\
/SecurityContext: \&v1.PodSecurityContext/{
N
N
s|.*\n\(.*\n\).*|\1|g
}"

find "${kubelet}" -type f -name *util.go -print0 | xargs -0 sed -i "\
/if pod.Spec.SecurityContext == nil {/ { N; N; N; d; }"

#===================
kuberuntime="${KUBE_ROOT}/pkg/kubelet/kuberuntime"
find "${kuberuntime}" -type f -name *.go -print0 | xargs -0 sed -i "s|\
\"true\"\.String()|\
\"true\"|g"

find "${kuberuntime}" -type f -name *.go -print0 | xargs -0 sed -i "s|\
string(uid)\.String()|\
string(uid)|g"

find "${kuberuntime}" -type f -name *.go -print0 | xargs -0 sed -i "s|\
string(podUID)\.String()|\
string(podUID)|g"

#podUID\.String()|\
#podUID|g"

find "${kuberuntime}" -type f -name *kuberuntime_sandbox.go -print0 | xargs -0 sed -i "\
/NamespaceOptions: \&runtimeApi\.NamespaceOption{/{
N
N
N
s|\(.*\)\n.*\n.*\n.*|\
\1\n\
    			HostNetwork: \&pod.Spec.HostNetwork,\n\
    			HostIpc:     \&pod.Spec.HostIPC,\n\
    			HostPid:     \&pod.Spec.HostPID,|g
}"

find "${kuberuntime}" -type f -name *security_context.go -print0 | xargs -0 sed -i "\
/runtimeapi.NamespaceOption/{
N
N
N
s|\(.*\)\n.*\n.*\n.*|\
\1\n\
    HostNetwork: \&pod.Spec.HostNetwork,\n\
    HostIpc:     \&pod.Spec.HostIPC,\n\
    HostPid:     \&pod.Spec.HostPID,|g
}"

#===================
server="${KUBE_ROOT}/pkg/kubelet/server"
find "${server}" -type f -name *server.go -print0 | xargs -0 sed -i "s|\
k8s.io/kubernetes/pkg/api/validation|\
k8s.io/kubernetes/pkg/api/v1/validation|g"


find "${server}/stats" -type f -name *summary_test.go -print0 | xargs -0 sed -i "s|\
\"k8s.io/kubernetes/pkg/api/v1\"|\
k8sv1 \"k8s.io/kubernetes/pkg/api/v1\"|g"

find "${server}/stats" -type f -name *summary_test.go -print0 | xargs -0 sed -i "s|\
v1.Node|\
k8sv1.Node|g"


#===================
config="${KUBE_ROOT}/pkg/kubelet/config"
find "${config}" -type f -name *config.go -print0 | xargs -0 sed -i "s|\
^\(.*\)validation.ValidatePod(pod)\(.*\)$|\
// TODO: remove the conversion when validation is performed on versioned objects.\n\
internalPod := \&api.Pod{}\n\
		if err := v1.Convert_v1_Pod_To_api_Pod(pod, internalPod, nil); err != nil {\n\
			name := kubecontainer.GetPodFullName(pod)\n\
			glog.Warningf(\"Pod[%d] (%s) from %s failed to convert to v1, ignoring: %v\", i+1, name, source, err)\n\
			recorder.Eventf(pod, v1.EventTypeWarning, \"FailedConversion\", \"Error converting pod %s from %s, ignoring: %v\", name, source, err)\n\
			continue\n\
		}\n\
        \1validation.ValidatePod(internalPod)\2|g"
#=====================================================================
####### static pods #########
find "${config}" -type f -name *file.go -print0 | xargs -0 sed -i "s|\
defaultFn := func(pod \*v1.Pod) error {|\
defaultFn := func(pod *api.Pod) error {|g"

find "${config}" -type f \( -name *http.go -o -name *file.go \) -print0 | xargs -0 sed -i "s|\
applyDefaults(pod \*v1.Pod|\
applyDefaults(pod *api.Pod|g"

find "${config}" -type f \( -name *http.go -o -name *file.go \) -print0 | xargs -0 sed -i "s|\
\"k8s.io/kubernetes/pkg/api/v1\"|\
\"k8s.io/kubernetes/pkg/api\"\n\
\"k8s.io/kubernetes/pkg/api/v1\"|g"
#===================
#common.go
find "${config}" -type f -name *common.go -print0 | xargs -0 sed -i "s|\
\"k8s.io/kubernetes/pkg/api\"|\
\"k8s.io/kubernetes/pkg/api\"\n\
\"k8s.io/kubernetes/pkg/api/v1\"|g"

find "${config}" -type f -name *common.go -print0 | xargs -0 sed -i "s|\
func tryDecodeSinglePod(data \[\]byte, defaultFn defaultFunc) (parsed bool, pod \*api.Pod, err error) {|\
func tryDecodeSinglePod(data \[\]byte, defaultFn defaultFunc) (parsed bool, pod *v1.Pod, err error) {|g"


find "${config}" -type f -name *common.go -print0 | xargs -0 sed -i "s|\
return true, newPod, nil|\
v1Pod := \&v1.Pod{}\n\
	if err := v1.Convert_api_Pod_To_v1_Pod(newPod, v1Pod, nil); err != nil {\n\
		return true, nil, err\n\
	}\n\
	return true, v1Pod, nil|g"


find "${config}" -type f -name *common.go -print0 | xargs -0 sed -i "s|\
func tryDecodePodList(data \[\]byte, defaultFn defaultFunc) (parsed bool, pods api.PodList, err error) {|\
func tryDecodePodList(data \[\]byte, defaultFn defaultFunc) (parsed bool, pods v1.PodList, err error) {|g"

find "${config}" -type f -name *common.go -print0 | xargs -0 sed -i "s|\
return true, \*newPods, err|\
v1Pods := \&v1.PodList{}\n\
	if err := v1.Convert_api_PodList_To_v1_PodList(newPods, v1Pods, nil); err != nil {\n\
		return true, pods, err\n\
	}\n\
	return true, *v1Pods, err|g"

#=============== tests
find "${config}" -type f -name *common_test.go -print0 | xargs -0 sed -i "s|\
noDefault(\*v1.Pod)|\
noDefault(*api.Pod)|g"

find "${config}" -type f \( -name *file_linux_test.go -o -name *http_test.go \) -print0 | xargs -0 sed -i "s|\
if errs := validation.ValidatePod(pod)\(.*\)$|\
// TODO: remove the conversion when validation is performed on versioned objects.\n\
				internalPod := \&api.Pod{}\n\
				if err := v1.Convert_v1_Pod_To_api_Pod(pod, internalPod, nil); err != nil {\n\
					t.Fatalf(\"%s: Cannot convert pod %#v, %#v\", testCase.desc, pod, err)\n\
				}\n\
				if errs := validation.ValidatePod(internalPod)\1|g"

find "${config}" -type f -name *file_linux_test.go -print0 | xargs -0 sed -i "s|\
Invalid pod\(.*\), pod|\
Invalid pod\1, internalPod|g"

find "${config}" -type f -name *file_linux_test.go -print0 | xargs -0 sed -i "s|\
no validation errors\(.*\), pod|\
no validation errors\1, internalPod|g"

find "${config}" -type f \( -name *file_linux_test.go -o -name *http_test.go \) -print0 | xargs -0 sed -i "s|\
k8s.io/kubernetes/pkg/api/v1\"|\
k8s.io/kubernetes/pkg/api\"\n\"k8s.io/kubernetes/pkg/api/v1\"|g"

#=====================================================================
securitycontext_fake="${KUBE_ROOT}/pkg/securitycontext/fake.go"
echo "
// ValidInternalSecurityContextWithContainerDefaults creates a valid security context provider based on
// empty container defaults.  Used for testing.
func ValidInternalSecurityContextWithContainerDefaults() *api.SecurityContext {
	priv := false
	return &api.SecurityContext{
		Capabilities: &api.Capabilities{},
		Privileged:   &priv,
	}
}" >> "${securitycontext_fake}"

sed -i "s|\
k8s.io/kubernetes/pkg/api/v1\"|\
k8s.io/kubernetes/pkg/api\"\n\"k8s.io/kubernetes/pkg/api/v1\"|g" "${securitycontext_fake}"

#=========

