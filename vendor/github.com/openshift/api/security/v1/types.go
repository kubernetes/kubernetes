package v1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// AllowAllCapabilities can be used as a value for the
// SecurityContextConstraints.AllowAllCapabilities field and means that any
// capabilities are allowed to be requested.
var AllowAllCapabilities corev1.Capability = "*"

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// SecurityContextConstraints governs the ability to make requests that affect the SecurityContext
// that will be applied to a container.
// For historical reasons SCC was exposed under the core Kubernetes API group.
// That exposure is deprecated and will be removed in a future release - users
// should instead use the security.openshift.io group to manage
// SecurityContextConstraints.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=securitycontextconstraints,scope=Cluster
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/470
// +openshift:file-pattern=cvoRunLevel=0000_03,operatorName=config-operator,operatorOrdering=01
// +kubebuilder:printcolumn:name="Priv",type=string,JSONPath=.allowPrivilegedContainer,description="Determines if a container can request to be run as privileged"
// +kubebuilder:printcolumn:name="Caps",type=string,JSONPath=.allowedCapabilities,description="A list of capabilities that can be requested to add to the container"
// +kubebuilder:printcolumn:name="SELinux",type=string,JSONPath=.seLinuxContext.type,description="Strategy that will dictate what labels will be set in the SecurityContext"
// +kubebuilder:printcolumn:name="RunAsUser",type=string,JSONPath=.runAsUser.type,description="Strategy that will dictate what RunAsUser is used in the SecurityContext"
// +kubebuilder:printcolumn:name="FSGroup",type=string,JSONPath=.fsGroup.type,description="Strategy that will dictate what fs group is used by the SecurityContext"
// +kubebuilder:printcolumn:name="SupGroup",type=string,JSONPath=.supplementalGroups.type,description="Strategy that will dictate what supplemental groups are used by the SecurityContext"
// +kubebuilder:printcolumn:name="Priority",type=string,JSONPath=.priority,description="Sort order of SCCs"
// +kubebuilder:printcolumn:name="ReadOnlyRootFS",type=string,JSONPath=.readOnlyRootFilesystem,description="Force containers to run with a read only root file system"
// +kubebuilder:printcolumn:name="Volumes",type=string,JSONPath=.volumes,description="White list of allowed volume plugins"
// +kubebuilder:singular=securitycontextconstraint
// +openshift:compatibility-gen:level=1
// +kubebuilder:metadata:annotations=release.openshift.io/bootstrap-required=true
type SecurityContextConstraints struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// priority influences the sort order of SCCs when evaluating which SCCs to try first for
	// a given pod request based on access in the Users and Groups fields.  The higher the int, the
	// higher priority. An unset value is considered a 0 priority. If scores
	// for multiple SCCs are equal they will be sorted from most restrictive to
	// least restrictive. If both priorities and restrictions are equal the
	// SCCs will be sorted by name.
	// +nullable
	Priority *int32 `json:"priority" protobuf:"varint,2,opt,name=priority"`

	// allowPrivilegedContainer determines if a container can request to be run as privileged.
	AllowPrivilegedContainer bool `json:"allowPrivilegedContainer" protobuf:"varint,3,opt,name=allowPrivilegedContainer"`
	// defaultAddCapabilities is the default set of capabilities that will be added to the container
	// unless the pod spec specifically drops the capability.  You may not list a capabiility in both
	// DefaultAddCapabilities and RequiredDropCapabilities.
	// +nullable
	// +listType=atomic
	DefaultAddCapabilities []corev1.Capability `json:"defaultAddCapabilities" protobuf:"bytes,4,rep,name=defaultAddCapabilities,casttype=Capability"`
	// requiredDropCapabilities are the capabilities that will be dropped from the container.  These
	// are required to be dropped and cannot be added.
	// +nullable
	// +listType=atomic
	RequiredDropCapabilities []corev1.Capability `json:"requiredDropCapabilities" protobuf:"bytes,5,rep,name=requiredDropCapabilities,casttype=Capability"`
	// allowedCapabilities is a list of capabilities that can be requested to add to the container.
	// Capabilities in this field maybe added at the pod author's discretion.
	// You must not list a capability in both AllowedCapabilities and RequiredDropCapabilities.
	// To allow all capabilities you may use '*'.
	// +nullable
	// +listType=atomic
	AllowedCapabilities []corev1.Capability `json:"allowedCapabilities" protobuf:"bytes,6,rep,name=allowedCapabilities,casttype=Capability"`
	// allowHostDirVolumePlugin determines if the policy allow containers to use the HostDir volume plugin
	// +k8s:conversion-gen=false
	AllowHostDirVolumePlugin bool `json:"allowHostDirVolumePlugin" protobuf:"varint,7,opt,name=allowHostDirVolumePlugin"`
	// volumes is a white list of allowed volume plugins.  FSType corresponds directly with the field names
	// of a VolumeSource (azureFile, configMap, emptyDir).  To allow all volumes you may use "*".
	// To allow no volumes, set to ["none"].
	// +nullable
	// +listType=atomic
	Volumes []FSType `json:"volumes" protobuf:"bytes,8,rep,name=volumes,casttype=FSType"`
	// allowedFlexVolumes is a whitelist of allowed Flexvolumes.  Empty or nil indicates that all
	// Flexvolumes may be used.  This parameter is effective only when the usage of the Flexvolumes
	// is allowed in the "Volumes" field.
	// +optional
	// +nullable
	// +listType=atomic
	AllowedFlexVolumes []AllowedFlexVolume `json:"allowedFlexVolumes,omitempty" protobuf:"bytes,21,rep,name=allowedFlexVolumes"`
	// allowHostNetwork determines if the policy allows the use of HostNetwork in the pod spec.
	AllowHostNetwork bool `json:"allowHostNetwork" protobuf:"varint,9,opt,name=allowHostNetwork"`
	// allowHostPorts determines if the policy allows host ports in the containers.
	AllowHostPorts bool `json:"allowHostPorts" protobuf:"varint,10,opt,name=allowHostPorts"`
	// allowHostPID determines if the policy allows host pid in the containers.
	AllowHostPID bool `json:"allowHostPID" protobuf:"varint,11,opt,name=allowHostPID"`
	// allowHostIPC determines if the policy allows host ipc in the containers.
	AllowHostIPC bool `json:"allowHostIPC" protobuf:"varint,12,opt,name=allowHostIPC"`
	// userNamespaceLevel determines if the policy allows host users in containers.
	// Valid values are "AllowHostLevel", "RequirePodLevel", and omitted.
	// When "AllowHostLevel" is set, a pod author may set `hostUsers` to either `true` or `false`.
	// When "RequirePodLevel" is set, a pod author must set `hostUsers` to `false`.
	// When omitted, the default value is "AllowHostLevel".
	// +openshift:enable:FeatureGate=UserNamespacesPodSecurityStandards
	// +kubebuilder:validation:Enum="AllowHostLevel";"RequirePodLevel"
	// +kubebuilder:default:="AllowHostLevel"
	// +default="AllowHostLevel"
	// +optional
	UserNamespaceLevel NamespaceLevelType `json:"userNamespaceLevel,omitempty" protobuf:"bytes,26,opt,name=userNamespaceLevel"`
	// defaultAllowPrivilegeEscalation controls the default setting for whether a
	// process can gain more privileges than its parent process.
	// +optional
	// +nullable
	DefaultAllowPrivilegeEscalation *bool `json:"defaultAllowPrivilegeEscalation,omitempty" protobuf:"varint,22,rep,name=defaultAllowPrivilegeEscalation"`
	// allowPrivilegeEscalation determines if a pod can request to allow
	// privilege escalation. If unspecified, defaults to true.
	// +optional
	// +nullable
	AllowPrivilegeEscalation *bool `json:"allowPrivilegeEscalation,omitempty" protobuf:"varint,23,rep,name=allowPrivilegeEscalation"`
	// seLinuxContext is the strategy that will dictate what labels will be set in the SecurityContext.
	// +nullable
	SELinuxContext SELinuxContextStrategyOptions `json:"seLinuxContext,omitempty" protobuf:"bytes,13,opt,name=seLinuxContext"`
	// runAsUser is the strategy that will dictate what RunAsUser is used in the SecurityContext.
	// +nullable
	RunAsUser RunAsUserStrategyOptions `json:"runAsUser,omitempty" protobuf:"bytes,14,opt,name=runAsUser"`
	// supplementalGroups is the strategy that will dictate what supplemental groups are used by the SecurityContext.
	// +nullable
	SupplementalGroups SupplementalGroupsStrategyOptions `json:"supplementalGroups,omitempty" protobuf:"bytes,15,opt,name=supplementalGroups"`
	// fsGroup is the strategy that will dictate what fs group is used by the SecurityContext.
	// +nullable
	FSGroup FSGroupStrategyOptions `json:"fsGroup,omitempty" protobuf:"bytes,16,opt,name=fsGroup"`
	// readOnlyRootFilesystem when set to true will force containers to run with a read only root file
	// system.  If the container specifically requests to run with a non-read only root file system
	// the SCC should deny the pod.
	// If set to false the container may run with a read only root file system if it wishes but it
	// will not be forced to.
	ReadOnlyRootFilesystem bool `json:"readOnlyRootFilesystem" protobuf:"varint,17,opt,name=readOnlyRootFilesystem"`

	// The users who have permissions to use this security context constraints
	// +optional
	// +nullable
	// +listType=atomic
	Users []string `json:"users" protobuf:"bytes,18,rep,name=users"`
	// The groups that have permission to use this security context constraints
	// +optional
	// +nullable
	// +listType=atomic
	Groups []string `json:"groups" protobuf:"bytes,19,rep,name=groups"`

	// seccompProfiles lists the allowed profiles that may be set for the pod or
	// container's seccomp annotations.  An unset (nil) or empty value means that no profiles may
	// be specifid by the pod or container.	The wildcard '*' may be used to allow all profiles.  When
	// used to generate a value for a pod the first non-wildcard profile will be used as
	// the default.
	// +nullable
	// +listType=atomic
	SeccompProfiles []string `json:"seccompProfiles,omitempty" protobuf:"bytes,20,opt,name=seccompProfiles"`

	// allowedUnsafeSysctls is a list of explicitly allowed unsafe sysctls, defaults to none.
	// Each entry is either a plain sysctl name or ends in "*" in which case it is considered
	// as a prefix of allowed sysctls. Single * means all unsafe sysctls are allowed.
	// Kubelet has to whitelist all allowed unsafe sysctls explicitly to avoid rejection.
	//
	// Examples:
	// e.g. "foo/*" allows "foo/bar", "foo/baz", etc.
	// e.g. "foo.*" allows "foo.bar", "foo.baz", etc.
	// +optional
	// +nullable
	// +listType=atomic
	AllowedUnsafeSysctls []string `json:"allowedUnsafeSysctls,omitempty" protobuf:"bytes,24,rep,name=allowedUnsafeSysctls"`
	// forbiddenSysctls is a list of explicitly forbidden sysctls, defaults to none.
	// Each entry is either a plain sysctl name or ends in "*" in which case it is considered
	// as a prefix of forbidden sysctls. Single * means all sysctls are forbidden.
	//
	// Examples:
	// e.g. "foo/*" forbids "foo/bar", "foo/baz", etc.
	// e.g. "foo.*" forbids "foo.bar", "foo.baz", etc.
	// +optional
	// +nullable
	// +listType=atomic
	ForbiddenSysctls []string `json:"forbiddenSysctls,omitempty" protobuf:"bytes,25,rep,name=forbiddenSysctls"`
}

// FS Type gives strong typing to different file systems that are used by volumes.
type FSType string

var (
	FSTypeAzureFile             FSType = "azureFile"
	FSTypeAzureDisk             FSType = "azureDisk"
	FSTypeFlocker               FSType = "flocker"
	FSTypeFlexVolume            FSType = "flexVolume"
	FSTypeHostPath              FSType = "hostPath"
	FSTypeEmptyDir              FSType = "emptyDir"
	FSTypeGCEPersistentDisk     FSType = "gcePersistentDisk"
	FSTypeAWSElasticBlockStore  FSType = "awsElasticBlockStore"
	FSTypeGitRepo               FSType = "gitRepo"
	FSTypeSecret                FSType = "secret"
	FSTypeNFS                   FSType = "nfs"
	FSTypeISCSI                 FSType = "iscsi"
	FSTypeGlusterfs             FSType = "glusterfs"
	FSTypePersistentVolumeClaim FSType = "persistentVolumeClaim"
	FSTypeRBD                   FSType = "rbd"
	FSTypeCinder                FSType = "cinder"
	FSTypeCephFS                FSType = "cephFS"
	FSTypeDownwardAPI           FSType = "downwardAPI"
	FSTypeFC                    FSType = "fc"
	FSTypeConfigMap             FSType = "configMap"
	FSTypeVsphereVolume         FSType = "vsphere"
	FSTypeQuobyte               FSType = "quobyte"
	FSTypePhotonPersistentDisk  FSType = "photonPersistentDisk"
	FSProjected                 FSType = "projected"
	FSPortworxVolume            FSType = "portworxVolume"
	FSScaleIO                   FSType = "scaleIO"
	FSStorageOS                 FSType = "storageOS"
	FSTypeCSI                   FSType = "csi"
	FSTypeEphemeral             FSType = "ephemeral"
	FSTypeImage                 FSType = "image"
	FSTypeAll                   FSType = "*"
	FSTypeNone                  FSType = "none"
)

// AllowedFlexVolume represents a single Flexvolume that is allowed to be used.
type AllowedFlexVolume struct {
	// driver is the name of the Flexvolume driver.
	Driver string `json:"driver" protobuf:"bytes,1,opt,name=driver"`
}

// SELinuxContextStrategyOptions defines the strategy type and any options used to create the strategy.
type SELinuxContextStrategyOptions struct {
	// type is the strategy that will dictate what SELinux context is used in the SecurityContext.
	Type SELinuxContextStrategyType `json:"type,omitempty" protobuf:"bytes,1,opt,name=type,casttype=SELinuxContextStrategyType"`
	// seLinuxOptions required to run as; required for MustRunAs
	SELinuxOptions *corev1.SELinuxOptions `json:"seLinuxOptions,omitempty" protobuf:"bytes,2,opt,name=seLinuxOptions"`
}

// RunAsUserStrategyOptions defines the strategy type and any options used to create the strategy.
type RunAsUserStrategyOptions struct {
	// type is the strategy that will dictate what RunAsUser is used in the SecurityContext.
	Type RunAsUserStrategyType `json:"type,omitempty" protobuf:"bytes,1,opt,name=type,casttype=RunAsUserStrategyType"`
	// uid is the user id that containers must run as.  Required for the MustRunAs strategy if not using
	// namespace/service account allocated uids.
	UID *int64 `json:"uid,omitempty" protobuf:"varint,2,opt,name=uid"`
	// uidRangeMin defines the min value for a strategy that allocates by range.
	UIDRangeMin *int64 `json:"uidRangeMin,omitempty" protobuf:"varint,3,opt,name=uidRangeMin"`
	// uidRangeMax defines the max value for a strategy that allocates by range.
	UIDRangeMax *int64 `json:"uidRangeMax,omitempty" protobuf:"varint,4,opt,name=uidRangeMax"`
}

// FSGroupStrategyOptions defines the strategy type and options used to create the strategy.
type FSGroupStrategyOptions struct {
	// type is the strategy that will dictate what FSGroup is used in the SecurityContext.
	Type FSGroupStrategyType `json:"type,omitempty" protobuf:"bytes,1,opt,name=type,casttype=FSGroupStrategyType"`
	// ranges are the allowed ranges of fs groups.  If you would like to force a single
	// fs group then supply a single range with the same start and end.
	// +listType=atomic
	Ranges []IDRange `json:"ranges,omitempty" protobuf:"bytes,2,rep,name=ranges"`
}

// SupplementalGroupsStrategyOptions defines the strategy type and options used to create the strategy.
type SupplementalGroupsStrategyOptions struct {
	// type is the strategy that will dictate what supplemental groups is used in the SecurityContext.
	Type SupplementalGroupsStrategyType `json:"type,omitempty" protobuf:"bytes,1,opt,name=type,casttype=SupplementalGroupsStrategyType"`
	// ranges are the allowed ranges of supplemental groups.  If you would like to force a single
	// supplemental group then supply a single range with the same start and end.
	// +listType=atomic
	Ranges []IDRange `json:"ranges,omitempty" protobuf:"bytes,2,rep,name=ranges"`
}

// IDRange provides a min/max of an allowed range of IDs.
// TODO: this could be reused for UIDs.
type IDRange struct {
	// min is the start of the range, inclusive.
	Min int64 `json:"min,omitempty" protobuf:"varint,1,opt,name=min"`
	// max is the end of the range, inclusive.
	Max int64 `json:"max,omitempty" protobuf:"varint,2,opt,name=max"`
}

// NamespaceLevelType shows the allowable values for the UserNamespaceLevel field.
type NamespaceLevelType string

// SELinuxContextStrategyType denotes strategy types for generating SELinux options for a
// SecurityContext
type SELinuxContextStrategyType string

// RunAsUserStrategyType denotes strategy types for generating RunAsUser values for a
// SecurityContext
type RunAsUserStrategyType string

// SupplementalGroupsStrategyType denotes strategy types for determining valid supplemental
// groups for a SecurityContext.
type SupplementalGroupsStrategyType string

// FSGroupStrategyType denotes strategy types for generating FSGroup values for a
// SecurityContext
type FSGroupStrategyType string

const (
	// NamespaceLevelAllowHost allows a pod to set `hostUsers` field to either `true` or `false`
	NamespaceLevelAllowHost NamespaceLevelType = "AllowHostLevel"
	// NamespaceLevelRequirePod requires the `hostUsers` field be `false` in a pod.
	NamespaceLevelRequirePod NamespaceLevelType = "RequirePodLevel"

	// container must have SELinux labels of X applied.
	SELinuxStrategyMustRunAs SELinuxContextStrategyType = "MustRunAs"
	// container may make requests for any SELinux context labels.
	SELinuxStrategyRunAsAny SELinuxContextStrategyType = "RunAsAny"

	// container must run as a particular uid.
	RunAsUserStrategyMustRunAs RunAsUserStrategyType = "MustRunAs"
	// container must run as a particular uid.
	RunAsUserStrategyMustRunAsRange RunAsUserStrategyType = "MustRunAsRange"
	// container must run as a non-root uid
	RunAsUserStrategyMustRunAsNonRoot RunAsUserStrategyType = "MustRunAsNonRoot"
	// container may make requests for any uid.
	RunAsUserStrategyRunAsAny RunAsUserStrategyType = "RunAsAny"

	// container must have FSGroup of X applied.
	FSGroupStrategyMustRunAs FSGroupStrategyType = "MustRunAs"
	// container may make requests for any FSGroup labels.
	FSGroupStrategyRunAsAny FSGroupStrategyType = "RunAsAny"

	// container must run as a particular gid.
	SupplementalGroupsStrategyMustRunAs SupplementalGroupsStrategyType = "MustRunAs"
	// container may make requests for any gid.
	SupplementalGroupsStrategyRunAsAny SupplementalGroupsStrategyType = "RunAsAny"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// SecurityContextConstraintsList is a list of SecurityContextConstraints objects
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type SecurityContextConstraintsList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// List of security context constraints.
	Items []SecurityContextConstraints `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +genclient
// +genclient:onlyVerbs=create
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodSecurityPolicySubjectReview checks whether a particular user/SA tuple can create the PodTemplateSpec.
//
// Compatibility level 2: Stable within a major release for a minimum of 9 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=2
type PodSecurityPolicySubjectReview struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,3,opt,name=metadata"`

	// spec defines specification for the PodSecurityPolicySubjectReview.
	Spec PodSecurityPolicySubjectReviewSpec `json:"spec" protobuf:"bytes,1,opt,name=spec"`

	// status represents the current information/status for the PodSecurityPolicySubjectReview.
	Status PodSecurityPolicySubjectReviewStatus `json:"status,omitempty" protobuf:"bytes,2,opt,name=status"`
}

// PodSecurityPolicySubjectReviewSpec defines specification for PodSecurityPolicySubjectReview
type PodSecurityPolicySubjectReviewSpec struct {
	// template is the PodTemplateSpec to check. If template.spec.serviceAccountName is empty it will not be defaulted.
	// If its non-empty, it will be checked.
	Template corev1.PodTemplateSpec `json:"template" protobuf:"bytes,1,opt,name=template"`

	// user is the user you're testing for.
	// If you specify "user" but not "group", then is it interpreted as "What if user were not a member of any groups.
	// If user and groups are empty, then the check is performed using *only* the serviceAccountName in the template.
	User string `json:"user,omitempty" protobuf:"bytes,2,opt,name=user"`

	// groups is the groups you're testing for.
	Groups []string `json:"groups,omitempty" protobuf:"bytes,3,rep,name=groups"`
}

// PodSecurityPolicySubjectReviewStatus contains information/status for PodSecurityPolicySubjectReview.
type PodSecurityPolicySubjectReviewStatus struct {
	// allowedBy is a reference to the rule that allows the PodTemplateSpec.
	// A rule can be a SecurityContextConstraint or a PodSecurityPolicy
	// A `nil`, indicates that it was denied.
	// +optional
	AllowedBy *corev1.ObjectReference `json:"allowedBy,omitempty" protobuf:"bytes,1,opt,name=allowedBy"`

	// A machine-readable description of why this operation is in the
	// "Failure" status. If this value is empty there
	// is no information available.
	// +optional
	Reason string `json:"reason,omitempty" protobuf:"bytes,2,opt,name=reason"`

	// template is the PodTemplateSpec after the defaulting is applied.
	// +optional
	Template corev1.PodTemplateSpec `json:"template,omitempty" protobuf:"bytes,3,opt,name=template"`
}

// +genclient
// +genclient:onlyVerbs=create
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodSecurityPolicySelfSubjectReview checks whether this user/SA tuple can create the PodTemplateSpec
//
// Compatibility level 2: Stable within a major release for a minimum of 9 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=2
type PodSecurityPolicySelfSubjectReview struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,3,opt,name=metadata"`

	// spec defines specification the PodSecurityPolicySelfSubjectReview.
	Spec PodSecurityPolicySelfSubjectReviewSpec `json:"spec" protobuf:"bytes,1,opt,name=spec"`

	// status represents the current information/status for the PodSecurityPolicySelfSubjectReview.
	Status PodSecurityPolicySubjectReviewStatus `json:"status,omitempty" protobuf:"bytes,2,opt,name=status"`
}

// PodSecurityPolicySelfSubjectReviewSpec contains specification for PodSecurityPolicySelfSubjectReview.
type PodSecurityPolicySelfSubjectReviewSpec struct {
	// template is the PodTemplateSpec to check.
	Template corev1.PodTemplateSpec `json:"template" protobuf:"bytes,1,opt,name=template"`
}

// +genclient
// +genclient:onlyVerbs=create
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodSecurityPolicyReview checks which service accounts (not users, since that would be cluster-wide) can create the `PodTemplateSpec` in question.
//
// Compatibility level 2: Stable within a major release for a minimum of 9 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=2
type PodSecurityPolicyReview struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,3,opt,name=metadata"`

	// spec is the PodSecurityPolicy to check.
	Spec PodSecurityPolicyReviewSpec `json:"spec" protobuf:"bytes,1,opt,name=spec"`

	// status represents the current information/status for the PodSecurityPolicyReview.
	Status PodSecurityPolicyReviewStatus `json:"status,omitempty" protobuf:"bytes,2,opt,name=status"`
}

// PodSecurityPolicyReviewSpec defines specification for PodSecurityPolicyReview
type PodSecurityPolicyReviewSpec struct {
	// template is the PodTemplateSpec to check. The template.spec.serviceAccountName field is used
	// if serviceAccountNames is empty, unless the template.spec.serviceAccountName is empty,
	// in which case "default" is used.
	// If serviceAccountNames is specified, template.spec.serviceAccountName is ignored.
	Template corev1.PodTemplateSpec `json:"template" protobuf:"bytes,1,opt,name=template"`

	// serviceAccountNames is an optional set of ServiceAccounts to run the check with.
	// If serviceAccountNames is empty, the template.spec.serviceAccountName is used,
	// unless it's empty, in which case "default" is used instead.
	// If serviceAccountNames is specified, template.spec.serviceAccountName is ignored.
	ServiceAccountNames []string `json:"serviceAccountNames,omitempty" protobuf:"bytes,2,rep,name=serviceAccountNames"` // TODO: find a way to express 'all service accounts'
}

// PodSecurityPolicyReviewStatus represents the status of PodSecurityPolicyReview.
type PodSecurityPolicyReviewStatus struct {
	// allowedServiceAccounts returns the list of service accounts in *this* namespace that have the power to create the PodTemplateSpec.
	// +optional
	AllowedServiceAccounts []ServiceAccountPodSecurityPolicyReviewStatus `json:"allowedServiceAccounts" protobuf:"bytes,1,rep,name=allowedServiceAccounts"`
}

// ServiceAccountPodSecurityPolicyReviewStatus represents ServiceAccount name and related review status
type ServiceAccountPodSecurityPolicyReviewStatus struct {
	PodSecurityPolicySubjectReviewStatus `json:",inline" protobuf:"bytes,1,opt,name=podSecurityPolicySubjectReviewStatus"`

	// name contains the allowed and the denied ServiceAccount name
	Name string `json:"name" protobuf:"bytes,2,opt,name=name"`
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// RangeAllocation is used so we can easily expose a RangeAllocation typed for security group
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
type RangeAllocation struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// range is a string representing a unique label for a range of uids, "1000000000-2000000000/10000".
	Range string `json:"range" protobuf:"bytes,2,opt,name=range"`

	// data is a byte array representing the serialized state of a range allocation.  It is a bitmap
	// with each bit set to one to represent a range is taken.
	Data []byte `json:"data" protobuf:"bytes,3,opt,name=data"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// RangeAllocationList is a list of RangeAllocations objects
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type RangeAllocationList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// List of RangeAllocations.
	Items []RangeAllocation `json:"items" protobuf:"bytes,2,rep,name=items"`
}
