package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Scheduler holds cluster-wide config information to run the Kubernetes Scheduler
// and influence its placement decisions. The canonical name for this config is `cluster`.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/470
// +openshift:file-pattern=cvoRunLevel=0000_10,operatorName=config-operator,operatorOrdering=01
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=schedulers,scope=Cluster
// +kubebuilder:subresource:status
// +kubebuilder:metadata:annotations=release.openshift.io/bootstrap-required=true
type Scheduler struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +required
	Spec SchedulerSpec `json:"spec"`
	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status SchedulerStatus `json:"status"`
}

type SchedulerSpec struct {
	// DEPRECATED: the scheduler Policy API has been deprecated and will be removed in a future release.
	// policy is a reference to a ConfigMap containing scheduler policy which has
	// user specified predicates and priorities. If this ConfigMap is not available
	// scheduler will default to use DefaultAlgorithmProvider.
	// The namespace for this configmap is openshift-config.
	// +optional
	Policy ConfigMapNameReference `json:"policy,omitempty"`
	// profile sets which scheduling profile should be set in order to configure scheduling
	// decisions for new pods.
	//
	// Valid values are "LowNodeUtilization", "HighNodeUtilization", "NoScoring"
	// Defaults to "LowNodeUtilization"
	// +optional
	Profile SchedulerProfile `json:"profile,omitempty"`
	// profileCustomizations contains configuration for modifying the default behavior of existing scheduler profiles.
	// Deprecated: no longer needed, since DRA is GA starting with 4.21, and
	// is enabled by' default in the cluster, this field will be removed in 4.24.
	// +openshift:enable:FeatureGate=HyperShiftOnlyDynamicResourceAllocation
	// +optional
	ProfileCustomizations ProfileCustomizations `json:"profileCustomizations"`
	// defaultNodeSelector helps set the cluster-wide default node selector to
	// restrict pod placement to specific nodes. This is applied to the pods
	// created in all namespaces and creates an intersection with any existing
	// nodeSelectors already set on a pod, additionally constraining that pod's selector.
	// For example,
	// defaultNodeSelector: "type=user-node,region=east" would set nodeSelector
	// field in pod spec to "type=user-node,region=east" to all pods created
	// in all namespaces. Namespaces having project-wide node selectors won't be
	// impacted even if this field is set. This adds an annotation section to
	// the namespace.
	// For example, if a new namespace is created with
	// node-selector='type=user-node,region=east',
	// the annotation openshift.io/node-selector: type=user-node,region=east
	// gets added to the project. When the openshift.io/node-selector annotation
	// is set on the project the value is used in preference to the value we are setting
	// for defaultNodeSelector field.
	// For instance,
	// openshift.io/node-selector: "type=user-node,region=west" means
	// that the default of "type=user-node,region=east" set in defaultNodeSelector
	// would not be applied.
	// +optional
	DefaultNodeSelector string `json:"defaultNodeSelector,omitempty"`
	// mastersSchedulable allows masters nodes to be schedulable. When this flag is
	// turned on, all the master nodes in the cluster will be made schedulable,
	// so that workload pods can run on them. The default value for this field is false,
	// meaning none of the master nodes are schedulable.
	// Important Note: Once the workload pods start running on the master nodes,
	// extreme care must be taken to ensure that cluster-critical control plane components
	// are not impacted.
	// Please turn on this field after doing due diligence.
	// +optional
	MastersSchedulable bool `json:"mastersSchedulable"`
}

// +kubebuilder:validation:Enum="";LowNodeUtilization;HighNodeUtilization;NoScoring
type SchedulerProfile string

var (
	// LowNodeUtililization is the default, and defines a scheduling profile which prefers to
	// spread pods evenly among nodes targeting low resource consumption on each node.
	LowNodeUtilization SchedulerProfile = "LowNodeUtilization"

	// HighNodeUtilization defines a scheduling profile which packs as many pods as possible onto
	// as few nodes as possible targeting a small node count but high resource usage on each node.
	HighNodeUtilization SchedulerProfile = "HighNodeUtilization"

	// NoScoring defines a scheduling profile which tries to provide lower-latency scheduling
	// at the expense of potentially less optimal pod placement decisions.
	NoScoring SchedulerProfile = "NoScoring"
)

// ProfileCustomizations contains various parameters for modifying the default behavior of certain profiles
type ProfileCustomizations struct {
	// dynamicResourceAllocation allows to enable or disable dynamic resource allocation within the scheduler.
	// Dynamic resource allocation is an API for requesting and sharing resources between pods and containers inside a pod.
	// Third-party resource drivers are responsible for tracking and allocating resources.
	// Different kinds of resources support arbitrary parameters for defining requirements and initialization.
	// Valid values are Enabled, Disabled and omitted.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default,
	// which is subject to change over time.
	// The current default is Disabled.
	// +optional
	DynamicResourceAllocation DRAEnablement `json:"dynamicResourceAllocation"`
}

// +kubebuilder:validation:Enum:="";"Enabled";"Disabled"
type DRAEnablement string

var (
	// DRAEnablementEnabled enables dynamic resource allocation feature
	DRAEnablementEnabled DRAEnablement = "Enabled"
	// DRAEnablementDisabled disables dynamic resource allocation feature
	DRAEnablementDisabled DRAEnablement = "Disabled"
)

type SchedulerStatus struct {
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type SchedulerList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []Scheduler `json:"items"`
}
