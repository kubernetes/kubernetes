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

        #-path './plugin/pkg/admission/*' \
        #-path './pkg/registry/rbac/rest/*' -o \
        #-path './plugin/pkg/auth/authorizer/rbac/*' \

find ./ -type f -name "*.go" \
    \( \
        -path './pkg/dns/*' \
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
apps \"k8s.io/kubernetes/pkg/apis/apps/v1alpha1\"\n appsinternal \"k8s.io/kubernetes/pkg/apis/apps\"|g" "${target}"

sed -i "s|\"k8s.io/kubernetes/pkg/apis/policy\"|\
policy \"k8s.io/kubernetes/pkg/apis/policy/v1alpha1\"\n policyinternal \"k8s.io/kubernetes/pkg/apis/policy\"|g" "${target}"

sed -i "s|\"k8s.io/kubernetes/pkg/apis/rbac\"|\
rbac \"k8s.io/kubernetes/pkg/apis/rbac/v1alpha1\"\n rbacinternal \"k8s.io/kubernetes/pkg/apis/rbac\"|g" "${target}"

sed -i "s|rbac rbac|rbac|g" "${target}"

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
apparmor_strategy="${KUBE_ROOT}/pkg/security/podsecuritypolicy/apparmor"
find "${apparmor_strategy}" -type f -name *strategy.go -print0 | xargs -0 sed -i "s|\
apparmor.GetProfileName(pod,|\
apparmor.GetProfileNameFromPodAnnotations(pod.Annotations,|g"

#=============================ADMISSION========================================
#======ListOptions in reflect watch cache
admission="${KUBE_ROOT}/plugin/pkg/admission"
find "${admission}" -type f -name *.go -print0 | xargs -0 sed -i "\
/ListFunc: func(options api.ListOptions) (runtime.Object, error) {/{
N
N
s|List(options)|List(internalOptions)|g
s|options.FieldSelector|internalOptions.FieldSelector|g
s|.*\n\(.*\)\n\(.*\)|\
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {\n\
				internalOptions := api.ListOptions{}\n\
				v1.Convert_v1_ListOptions_To_api_ListOptions(\&options, \&internalOptions, nil)\n\
                \1\n\
                \2|g
}"

find "${admission}" -type f -name *.go -print0 | xargs -0 sed -i "\
/WatchFunc: func(options api.ListOptions) (watch.Interface, error) {/{
N
N
s|Watch(options)|Watch(internalOptions)|g
s|options.FieldSelector|internalOptions.FieldSelector|g
s|.*\n\(.*\)\n\(.*\)|\
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {\n\
				internalOptions := api.ListOptions{}\n\
				v1.Convert_v1_ListOptions_To_api_ListOptions(\&options, \&internalOptions, nil)\n\
                \1\n\
                \2|g
}"

find "${admission}" -type f -name *.go -print0 | xargs -0 sed -i "s|\
\"k8s.io/kubernetes/pkg/api\"|\
\"k8s.io/kubernetes/pkg/api\"\n\
\"k8s.io/kubernetes/pkg/api/v1\"|g"

find "${admission}" -type f -name *.go -print0 | xargs -0 sed -i "s|\
NewSharedInformerFactory(c|\
NewSharedInformerFactory(nil, c|g"

find "${admission}" -type f -name *.go -print0 | xargs -0 sed -i "s|\
\.LimitRanges()\.Informer()|\
.InternalLimitRanges().Informer()|g"

find "${admission}" -type f -name *.go -print0 | xargs -0 sed -i "s|\
\.LimitRanges()\.Lister()|\
.InternalLimitRanges().Lister()|g"

find "${admission}" -type f -name *.go -print0 | xargs -0 sed -i "s|\
\.Namespaces()\.Informer()|\
.InternalNamespaces()\.Informer()|g"

find "${admission}" -type f -name *.go -print0 | xargs -0 sed -i "s|\
serviceaccount\.IsServiceAccountToken|\
serviceaccount.InternalIsServiceAccountToken|g"

find "${admission}/resourcequota" -type f -name *.go -print0 | xargs -0 sed -i "s|\
pkg/quota|\
pkg/quotainternal|g"

find "${admission}/imagepolicy" -type f -name *admission_test.go -print0 | xargs -0 sed -i \
"/\"k8s.io\/kubernetes\/pkg\/api\/v1\"/d"
    
goimports -w "${admission}"

#=========
echo "
// Sets the name of the profile to use with the container.
func SetProfileNameFromPodAnnotations(annotations map[string]string, containerName, profileName string) error {
	if annotations == nil {
		return nil
	}
	annotations[ContainerAnnotationKeyPrefix+containerName] = profileName
	return nil
}" >> "${KUBE_ROOT}"/pkg/security/apparmor/helpers.go 

podsecuritypolicy="${KUBE_ROOT}/plugin/pkg/admission/security/podsecuritypolicy"
find "${podsecuritypolicy}" -type f -name *admission_test.go -print0 | xargs -0 sed -i "s|\
apparmor.SetProfileName(pod|\
apparmor.SetProfileNameFromPodAnnotations(pod.Annotations|g"

find "${podsecuritypolicy}" -type f -name *admission_test.go -print0 | xargs -0 sed -i "s|\
apparmor.GetProfileName(v.pod|\
apparmor.GetProfileNameFromPodAnnotations(v.pod.Annotations|g"

find "${podsecuritypolicy}" -type f -name *admission.go -print0 | xargs -0 sed -i "s|\
DetermineEffectiveSecurityContext|\
InternalDetermineEffectiveSecurityContext|g"

#======== A lot...
echo "
// TODO: remove the duplicate code
func InternalDetermineEffectiveSecurityContext(pod *api.Pod, container *api.Container) *api.SecurityContext {
	effectiveSc := internalSecurityContextFromPodSecurityContext(pod)
	containerSc := container.SecurityContext

	if effectiveSc == nil && containerSc == nil {
		return nil
	}
	if effectiveSc != nil && containerSc == nil {
		return effectiveSc
	}
	if effectiveSc == nil && containerSc != nil {
		return containerSc
	}

	if containerSc.SELinuxOptions != nil {
		effectiveSc.SELinuxOptions = new(api.SELinuxOptions)
		*effectiveSc.SELinuxOptions = *containerSc.SELinuxOptions
	}

	if containerSc.Capabilities != nil {
		effectiveSc.Capabilities = new(api.Capabilities)
		*effectiveSc.Capabilities = *containerSc.Capabilities
	}

	if containerSc.Privileged != nil {
		effectiveSc.Privileged = new(bool)
		*effectiveSc.Privileged = *containerSc.Privileged
	}

	if containerSc.RunAsUser != nil {
		effectiveSc.RunAsUser = new(int64)
		*effectiveSc.RunAsUser = *containerSc.RunAsUser
	}

	if containerSc.RunAsNonRoot != nil {
		effectiveSc.RunAsNonRoot = new(bool)
		*effectiveSc.RunAsNonRoot = *containerSc.RunAsNonRoot
	}

	if containerSc.ReadOnlyRootFilesystem != nil {
		effectiveSc.ReadOnlyRootFilesystem = new(bool)
		*effectiveSc.ReadOnlyRootFilesystem = *containerSc.ReadOnlyRootFilesystem
	}

	return effectiveSc
}

func internalSecurityContextFromPodSecurityContext(pod *api.Pod) *api.SecurityContext {
	if pod.Spec.SecurityContext == nil {
		return nil
	}

	synthesized := &api.SecurityContext{}

	if pod.Spec.SecurityContext.SELinuxOptions != nil {
		synthesized.SELinuxOptions = &api.SELinuxOptions{}
		*synthesized.SELinuxOptions = *pod.Spec.SecurityContext.SELinuxOptions
	}
	if pod.Spec.SecurityContext.RunAsUser != nil {
		synthesized.RunAsUser = new(int64)
		*synthesized.RunAsUser = *pod.Spec.SecurityContext.RunAsUser
	}

	if pod.Spec.SecurityContext.RunAsNonRoot != nil {
		synthesized.RunAsNonRoot = new(bool)
		*synthesized.RunAsNonRoot = *pod.Spec.SecurityContext.RunAsNonRoot
	}

	return synthesized
}" >> "${KUBE_ROOT}/pkg/securitycontext/provider.go"

sed -i "s|\
\"k8s.io/kubernetes/pkg/api/v1\"|\
\"k8s.io/kubernetes/pkg/api/v1\"\n\
\"k8s.io/kubernetes/pkg/api\"|g" "${KUBE_ROOT}/pkg/securitycontext/provider.go" 


#==============================================generic apiserver=============
sed -i "s|\
api.NodeExternalIP|\
v1.NodeExternalIP|g" "${KUBE_ROOT}/pkg/genericapiserver/config.go"

sed -i "s|\
\"k8s.io/kubernetes/pkg/api\"|\
\"k8s.io/kubernetes/pkg/api/v1\"\n\
\"k8s.io/kubernetes/pkg/api\"|g" "${KUBE_ROOT}/pkg/genericapiserver/config.go"
