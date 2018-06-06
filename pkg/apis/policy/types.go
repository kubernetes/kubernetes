/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package policy

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// PodDisruptionBudgetSpec is a description of a PodDisruptionBudget.
type PodDisruptionBudgetSpec struct {
	// An eviction is allowed if at least "minAvailable" pods selected by
	// "selector" will still be available after the eviction, i.e. even in the
	// absence of the evicted pod.  So for example you can prevent all voluntary
	// evictions by specifying "100%".
	// +optional
	MinAvailable *intstr.IntOrString

	// Label query over pods whose evictions are managed by the disruption
	// budget.
	// +optional
	Selector *metav1.LabelSelector

	// An eviction is allowed if at most "maxUnavailable" pods selected by
	// "selector" are unavailable after the eviction, i.e. even in absence of
	// the evicted pod. For example, one can prevent all voluntary evictions
	// by specifying 0. This is a mutually exclusive setting with "minAvailable".
	// +optional
	MaxUnavailable *intstr.IntOrString
}

// PodDisruptionBudgetStatus represents information about the status of a
// PodDisruptionBudget. Status may trail the actual state of a system.
type PodDisruptionBudgetStatus struct {
	// Most recent generation observed when updating this PDB status. PodDisruptionsAllowed and other
	// status informatio is valid only if observedGeneration equals to PDB's object generation.
	// +optional
	ObservedGeneration int64

	// DisruptedPods contains information about pods whose eviction was
	// processed by the API server eviction subresource handler but has not
	// yet been observed by the PodDisruptionBudget controller.
	// A pod will be in this map from the time when the API server processed the
	// eviction request to the time when the pod is seen by PDB controller
	// as having been marked for deletion (or after a timeout). The key in the map is the name of the pod
	// and the value is the time when the API server processed the eviction request. If
	// the deletion didn't occur and a pod is still there it will be removed from
	// the list automatically by PodDisruptionBudget controller after some time.
	// If everything goes smooth this map should be empty for the most of the time.
	// Large number of entries in the map may indicate problems with pod deletions.
	DisruptedPods map[string]metav1.Time

	// Number of pod disruptions that are currently allowed.
	PodDisruptionsAllowed int32

	// current number of healthy pods
	CurrentHealthy int32

	// minimum desired number of healthy pods
	DesiredHealthy int32

	// total number of pods counted by this disruption budget
	ExpectedPods int32
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodDisruptionBudget is an object to define the max disruption that can be caused to a collection of pods
type PodDisruptionBudget struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Specification of the desired behavior of the PodDisruptionBudget.
	// +optional
	Spec PodDisruptionBudgetSpec
	// Most recently observed status of the PodDisruptionBudget.
	// +optional
	Status PodDisruptionBudgetStatus
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodDisruptionBudgetList is a collection of PodDisruptionBudgets.
type PodDisruptionBudgetList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta
	Items []PodDisruptionBudget
}

// +genclient
// +genclient:noVerbs
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Eviction evicts a pod from its node subject to certain policies and safety constraints.
// This is a subresource of Pod.  A request to cause such an eviction is
// created by POSTing to .../pods/<pod name>/eviction.
type Eviction struct {
	metav1.TypeMeta

	// ObjectMeta describes the pod that is being evicted.
	// +optional
	metav1.ObjectMeta

	// DeleteOptions may be provided
	// +optional
	DeleteOptions *metav1.DeleteOptions
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodSecurityPolicy governs the ability to make requests that affect the SecurityContext
// that will be applied to a pod and container.
type PodSecurityPolicy struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Spec defines the policy enforced.
	// +optional
	Spec PodSecurityPolicySpec
}

// PodSecurityPolicySpec defines the policy enforced.
type PodSecurityPolicySpec struct {
	// Privileged determines if a pod can request to be run as privileged.
	// +optional
	Privileged bool
	// DefaultAddCapabilities is the default set of capabilities that will be added to the container
	// unless the pod spec specifically drops the capability.  You may not list a capability in both
	// DefaultAddCapabilities and RequiredDropCapabilities. Capabilities added here are implicitly
	// allowed, and need not be included in the AllowedCapabilities list.
	// +optional
	DefaultAddCapabilities []api.Capability
	// RequiredDropCapabilities are the capabilities that will be dropped from the container.  These
	// are required to be dropped and cannot be added.
	// +optional
	RequiredDropCapabilities []api.Capability
	// AllowedCapabilities is a list of capabilities that can be requested to add to the container.
	// Capabilities in this field may be added at the pod author's discretion.
	// You must not list a capability in both AllowedCapabilities and RequiredDropCapabilities.
	// To allow all capabilities you may use '*'.
	// +optional
	AllowedCapabilities []api.Capability
	// Volumes is a white list of allowed volume plugins. Empty indicates that
	// no volumes may be used. To allow all volumes you may use '*'.
	// +optional
	Volumes []FSType
	// HostNetwork determines if the policy allows the use of HostNetwork in the pod spec.
	// +optional
	HostNetwork bool
	// HostPorts determines which host port ranges are allowed to be exposed.
	// +optional
	HostPorts []HostPortRange
	// HostPID determines if the policy allows the use of HostPID in the pod spec.
	// +optional
	HostPID bool
	// HostIPC determines if the policy allows the use of HostIPC in the pod spec.
	// +optional
	HostIPC bool
	// SELinux is the strategy that will dictate the allowable labels that may be set.
	SELinux SELinuxStrategyOptions
	// RunAsUser is the strategy that will dictate the allowable RunAsUser values that may be set.
	RunAsUser RunAsUserStrategyOptions
	// SupplementalGroups is the strategy that will dictate what supplemental groups are used by the SecurityContext.
	SupplementalGroups SupplementalGroupsStrategyOptions
	// FSGroup is the strategy that will dictate what fs group is used by the SecurityContext.
	FSGroup FSGroupStrategyOptions
	// ReadOnlyRootFilesystem when set to true will force containers to run with a read only root file
	// system.  If the container specifically requests to run with a non-read only root file system
	// the PSP should deny the pod.
	// If set to false the container may run with a read only root file system if it wishes but it
	// will not be forced to.
	// +optional
	ReadOnlyRootFilesystem bool
	// DefaultAllowPrivilegeEscalation controls the default setting for whether a
	// process can gain more privileges than its parent process.
	// +optional
	DefaultAllowPrivilegeEscalation *bool
	// AllowPrivilegeEscalation determines if a pod can request to allow
	// privilege escalation. If unspecified, defaults to true.
	// +optional
	AllowPrivilegeEscalation bool
	// AllowedHostPaths is a white list of allowed host paths. Empty indicates that all host paths may be used.
	// +optional
	AllowedHostPaths []AllowedHostPath
	// AllowedFlexVolumes is a whitelist of allowed Flexvolumes.  Empty or nil indicates that all
	// Flexvolumes may be used.  This parameter is effective only when the usage of the Flexvolumes
	// is allowed in the "Volumes" field.
	// +optional
	AllowedFlexVolumes []AllowedFlexVolume
	// AllowedUnsafeSysctls is a list of explicitly allowed unsafe sysctls, defaults to none.
	// Each entry is either a plain sysctl name or ends in "*" in which case it is considered
	// as a prefix of allowed sysctls. Single * means all unsafe sysctls are allowed.
	// Kubelet has to whitelist all allowed unsafe sysctls explicitly to avoid rejection.
	//
	// Examples:
	// e.g. "foo/*" allows "foo/bar", "foo/baz", etc.
	// e.g. "foo.*" allows "foo.bar", "foo.baz", etc.
	// +optional
	AllowedUnsafeSysctls []string
	// ForbiddenSysctls is a list of explicitly forbidden sysctls, defaults to none.
	// Each entry is either a plain sysctl name or ends in "*" in which case it is considered
	// as a prefix of forbidden sysctls. Single * means all sysctls are forbidden.
	//
	// Examples:
	// e.g. "foo/*" forbids "foo/bar", "foo/baz", etc.
	// e.g. "foo.*" forbids "foo.bar", "foo.baz", etc.
	// +optional
	ForbiddenSysctls []string
}

// AllowedHostPath defines the host volume conditions that will be enabled by a policy
// for pods to use. It requires the path prefix to be defined.
type AllowedHostPath struct {
	// PathPrefix is the path prefix that the host volume must match.
	// PathPrefix does not support `*`.
	// Trailing slashes are trimmed when validating the path prefix with a host path.
	//
	// Examples:
	// `/foo` would allow `/foo`, `/foo/` and `/foo/bar`
	// `/foo` would not allow `/food` or `/etc/foo`
	PathPrefix string

	// when set to true, will allow host volumes matching the pathPrefix only if all volume mounts are readOnly.
	ReadOnly bool
}

// HostPortRange defines a range of host ports that will be enabled by a policy
// for pods to use.  It requires both the start and end to be defined.
type HostPortRange struct {
	// Min is the start of the range, inclusive.
	Min int32
	// Max is the end of the range, inclusive.
	Max int32
}

// AllowAllCapabilities can be used as a value for the PodSecurityPolicy.AllowAllCapabilities
// field and means that any capabilities are allowed to be requested.
var AllowAllCapabilities api.Capability = "*"

// FSType gives strong typing to different file systems that are used by volumes.
type FSType string

var (
	AzureFile             FSType = "azureFile"
	Flocker               FSType = "flocker"
	FlexVolume            FSType = "flexVolume"
	HostPath              FSType = "hostPath"
	EmptyDir              FSType = "emptyDir"
	GCEPersistentDisk     FSType = "gcePersistentDisk"
	AWSElasticBlockStore  FSType = "awsElasticBlockStore"
	GitRepo               FSType = "gitRepo"
	Secret                FSType = "secret"
	NFS                   FSType = "nfs"
	ISCSI                 FSType = "iscsi"
	Glusterfs             FSType = "glusterfs"
	PersistentVolumeClaim FSType = "persistentVolumeClaim"
	RBD                   FSType = "rbd"
	Cinder                FSType = "cinder"
	CephFS                FSType = "cephFS"
	DownwardAPI           FSType = "downwardAPI"
	FC                    FSType = "fc"
	ConfigMap             FSType = "configMap"
	VsphereVolume         FSType = "vsphereVolume"
	Quobyte               FSType = "quobyte"
	AzureDisk             FSType = "azureDisk"
	PhotonPersistentDisk  FSType = "photonPersistentDisk"
	StorageOS             FSType = "storageos"
	Projected             FSType = "projected"
	PortworxVolume        FSType = "portworxVolume"
	ScaleIO               FSType = "scaleIO"
	CSI                   FSType = "csi"
	All                   FSType = "*"
)

// AllowedFlexVolume represents a single Flexvolume that is allowed to be used.
type AllowedFlexVolume struct {
	// Driver is the name of the Flexvolume driver.
	Driver string
}

// SELinuxStrategyOptions defines the strategy type and any options used to create the strategy.
type SELinuxStrategyOptions struct {
	// Rule is the strategy that will dictate the allowable labels that may be set.
	Rule SELinuxStrategy
	// SELinuxOptions required to run as; required for MustRunAs
	// More info: https://kubernetes.io/docs/concepts/policy/pod-security-policy/#selinux
	// +optional
	SELinuxOptions *api.SELinuxOptions
}

// SELinuxStrategy denotes strategy types for generating SELinux options for a
// Security.
type SELinuxStrategy string

const (
	// SELinuxStrategyMustRunAs means that container must have SELinux labels of X applied.
	SELinuxStrategyMustRunAs SELinuxStrategy = "MustRunAs"
	// SELinuxStrategyRunAsAny means that container may make requests for any SELinux context labels.
	SELinuxStrategyRunAsAny SELinuxStrategy = "RunAsAny"
)

// RunAsUserStrategyOptions defines the strategy type and any options used to create the strategy.
type RunAsUserStrategyOptions struct {
	// Rule is the strategy that will dictate the allowable RunAsUser values that may be set.
	Rule RunAsUserStrategy
	// Ranges are the allowed ranges of uids that may be used. If you would like to force a single uid
	// then supply a single range with the same start and end. Required for MustRunAs.
	// +optional
	Ranges []IDRange
}

// IDRange provides a min/max of an allowed range of IDs.
type IDRange struct {
	// Min is the start of the range, inclusive.
	Min int64
	// Max is the end of the range, inclusive.
	Max int64
}

// RunAsUserStrategy denotes strategy types for generating RunAsUser values for a
// SecurityContext.
type RunAsUserStrategy string

const (
	// RunAsUserStrategyMustRunAs means that container must run as a particular uid.
	RunAsUserStrategyMustRunAs RunAsUserStrategy = "MustRunAs"
	// RunAsUserStrategyMustRunAsNonRoot means that container must run as a non-root uid
	RunAsUserStrategyMustRunAsNonRoot RunAsUserStrategy = "MustRunAsNonRoot"
	// RunAsUserStrategyRunAsAny means that container may make requests for any uid.
	RunAsUserStrategyRunAsAny RunAsUserStrategy = "RunAsAny"
)

// FSGroupStrategyOptions defines the strategy type and options used to create the strategy.
type FSGroupStrategyOptions struct {
	// Rule is the strategy that will dictate what FSGroup is used in the SecurityContext.
	// +optional
	Rule FSGroupStrategyType
	// Ranges are the allowed ranges of fs groups.  If you would like to force a single
	// fs group then supply a single range with the same start and end. Required for MustRunAs.
	// +optional
	Ranges []IDRange
}

// FSGroupStrategyType denotes strategy types for generating FSGroup values for a
// SecurityContext
type FSGroupStrategyType string

const (
	// FSGroupStrategyMustRunAs means that container must have FSGroup of X applied.
	FSGroupStrategyMustRunAs FSGroupStrategyType = "MustRunAs"
	// FSGroupStrategyRunAsAny means that container may make requests for any FSGroup labels.
	FSGroupStrategyRunAsAny FSGroupStrategyType = "RunAsAny"
)

// SupplementalGroupsStrategyOptions defines the strategy type and options used to create the strategy.
type SupplementalGroupsStrategyOptions struct {
	// Rule is the strategy that will dictate what supplemental groups is used in the SecurityContext.
	// +optional
	Rule SupplementalGroupsStrategyType
	// Ranges are the allowed ranges of supplemental groups.  If you would like to force a single
	// supplemental group then supply a single range with the same start and end. Required for MustRunAs.
	// +optional
	Ranges []IDRange
}

// SupplementalGroupsStrategyType denotes strategy types for determining valid supplemental
// groups for a SecurityContext.
type SupplementalGroupsStrategyType string

const (
	// SupplementalGroupsStrategyMustRunAs means that container must run as a particular gid.
	SupplementalGroupsStrategyMustRunAs SupplementalGroupsStrategyType = "MustRunAs"
	// SupplementalGroupsStrategyRunAsAny means that container may make requests for any gid.
	SupplementalGroupsStrategyRunAsAny SupplementalGroupsStrategyType = "RunAsAny"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodSecurityPolicyList is a list of PodSecurityPolicy objects.
type PodSecurityPolicyList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []PodSecurityPolicy
}
