package v1

import (
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Node holds cluster-wide information about node specific features.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/1107
// +openshift:file-pattern=cvoRunLevel=0000_10,operatorName=config-operator,operatorOrdering=01
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=nodes,scope=Cluster
// +kubebuilder:subresource:status
// +kubebuilder:metadata:annotations=release.openshift.io/bootstrap-required=true
type Node struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +required
	Spec NodeSpec `json:"spec"`

	// status holds observed values.
	// +optional
	Status NodeStatus `json:"status"`
}

type NodeSpec struct {
	// cgroupMode determines the cgroups version on the node
	// +optional
	CgroupMode CgroupMode `json:"cgroupMode,omitempty"`

	// workerLatencyProfile determins the how fast the kubelet is updating
	// the status and corresponding reaction of the cluster
	// +optional
	WorkerLatencyProfile WorkerLatencyProfileType `json:"workerLatencyProfile,omitempty"`

	// minimumKubeletVersion is the lowest version of a kubelet that can join the cluster.
	// Specifically, the apiserver will deny most authorization requests of kubelets that are older
	// than the specified version, only allowing the kubelet to get and update its node object, and perform
	// subjectaccessreviews.
	// This means any kubelet that attempts to join the cluster will not be able to run any assigned workloads,
	// and will eventually be marked as not ready.
	// Its max length is 8, so maximum version allowed is either "9.999.99" or "99.99.99".
	// Since the kubelet reports the version of the kubernetes release, not Openshift, this field references
	// the underlying kubernetes version this version of Openshift is based off of.
	// In other words: if an admin wishes to ensure no nodes run an older version than Openshift 4.17, then
	// they should set the minimumKubeletVersion to 1.30.0.
	// When comparing versions, the kubelet's version is stripped of any contents outside of major.minor.patch version.
	// Thus, a kubelet with version "1.0.0-ec.0" will be compatible with minimumKubeletVersion "1.0.0" or earlier.
	// +kubebuilder:validation:XValidation:rule="self == \"\" || self.matches('^[0-9]*.[0-9]*.[0-9]*$')",message="minmumKubeletVersion must be in a semver compatible format of x.y.z, or empty"
	// +kubebuilder:validation:MaxLength:=8
	// +openshift:enable:FeatureGate=MinimumKubeletVersion
	// +optional
	MinimumKubeletVersion string `json:"minimumKubeletVersion"`
}

type NodeStatus struct {
	// conditions contain the details and the current state of the nodes.config object
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:validation:Enum=v2;""
type CgroupMode string

const (
	CgroupModeEmpty   CgroupMode = "" // Empty string indicates to honor user set value on the system that should not be overridden by OpenShift
	CgroupModeV1      CgroupMode = "v1"
	CgroupModeV2      CgroupMode = "v2"
	CgroupModeDefault CgroupMode = CgroupModeV2
)

// +kubebuilder:validation:Enum=Default;MediumUpdateAverageReaction;LowUpdateSlowReaction
type WorkerLatencyProfileType string

const (
	// Medium Kubelet Update Frequency (heart-beat) and Average Reaction Time to unresponsive Node
	MediumUpdateAverageReaction WorkerLatencyProfileType = "MediumUpdateAverageReaction"

	// Low Kubelet Update Frequency (heart-beat) and Slow Reaction Time to unresponsive Node
	LowUpdateSlowReaction WorkerLatencyProfileType = "LowUpdateSlowReaction"

	// Default values of relavent Kubelet, Kube Controller Manager and Kube API Server
	DefaultUpdateDefaultReaction WorkerLatencyProfileType = "Default"
)

const (
	// DefaultNodeStatusUpdateFrequency refers to the "--node-status-update-frequency" of the kubelet in case of DefaultUpdateDefaultReaction WorkerLatencyProfile type
	DefaultNodeStatusUpdateFrequency = 10 * time.Second
	// DefaultNodeMonitorGracePeriod refers to the "--node-monitor-grace-period" of the Kube Controller Manager in case of DefaultUpdateDefaultReaction WorkerLatencyProfile type
	DefaultNodeMonitorGracePeriod = 40 * time.Second
	// DefaultNotReadyTolerationSeconds refers to the "--default-not-ready-toleration-seconds" of the Kube API Server in case of DefaultUpdateDefaultReaction WorkerLatencyProfile type
	DefaultNotReadyTolerationSeconds = 300
	// DefaultUnreachableTolerationSeconds refers to the "--default-unreachable-toleration-seconds" of the Kube API Server in case of DefaultUpdateDefaultReaction WorkerLatencyProfile type
	DefaultUnreachableTolerationSeconds = 300

	// MediumNodeStatusUpdateFrequency refers to the "--node-status-update-frequency" of the kubelet in case of MediumUpdateAverageReaction WorkerLatencyProfile type
	MediumNodeStatusUpdateFrequency = 20 * time.Second
	// MediumNodeMonitorGracePeriod refers to the "--node-monitor-grace-period" of the Kube Controller Manager in case of MediumUpdateAverageReaction WorkerLatencyProfile type
	MediumNodeMonitorGracePeriod = 2 * time.Minute
	// MediumNotReadyTolerationSeconds refers to the "--default-not-ready-toleration-seconds" of the Kube API Server in case of MediumUpdateAverageReaction WorkerLatencyProfile type
	MediumNotReadyTolerationSeconds = 60
	// MediumUnreachableTolerationSeconds refers to the "--default-unreachable-toleration-seconds" of the Kube API Server in case of MediumUpdateAverageReaction WorkerLatencyProfile type
	MediumUnreachableTolerationSeconds = 60

	// LowNodeStatusUpdateFrequency refers to the "--node-status-update-frequency" of the kubelet in case of LowUpdateSlowReaction WorkerLatencyProfile type
	LowNodeStatusUpdateFrequency = 1 * time.Minute
	// LowNodeMonitorGracePeriod refers to the "--node-monitor-grace-period" of the Kube Controller Manager in case of LowUpdateSlowReaction WorkerLatencyProfile type
	LowNodeMonitorGracePeriod = 5 * time.Minute
	// LowNotReadyTolerationSeconds refers to the "--default-not-ready-toleration-seconds" of the Kube API Server in case of LowUpdateSlowReaction WorkerLatencyProfile type
	LowNotReadyTolerationSeconds = 60
	// LowUnreachableTolerationSeconds refers to the "--default-unreachable-toleration-seconds" of the Kube API Server in case of LowUpdateSlowReaction WorkerLatencyProfile type
	LowUnreachableTolerationSeconds = 60
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type NodeList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []Node `json:"items"`
}
