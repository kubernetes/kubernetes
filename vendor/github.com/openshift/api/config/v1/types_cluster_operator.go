package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ClusterOperator is the Custom Resource object which holds the current state
// of an operator. This object is used by operators to convey their state to
// the rest of the cluster.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/497
// +openshift:file-pattern=cvoRunLevel=0000_00,operatorName=cluster-version-operator,operatorOrdering=01
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=clusteroperators,scope=Cluster,shortName=co
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name=Version,JSONPath=.status.versions[?(@.name=="operator")].version,type=string,description=The version the operator is at.
// +kubebuilder:printcolumn:name=Available,JSONPath=.status.conditions[?(@.type=="Available")].status,type=string,description=Whether the operator is running and stable.
// +kubebuilder:printcolumn:name=Progressing,JSONPath=.status.conditions[?(@.type=="Progressing")].status,type=string,description=Whether the operator is processing changes.
// +kubebuilder:printcolumn:name=Degraded,JSONPath=.status.conditions[?(@.type=="Degraded")].status,type=string,description=Whether the operator is degraded.
// +kubebuilder:printcolumn:name=Since,JSONPath=.status.conditions[?(@.type=="Available")].lastTransitionTime,type=date,description=The time the operator's Available status last changed.
// +kubebuilder:metadata:annotations=include.release.openshift.io/self-managed-high-availability=true
type ClusterOperator struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata"`

	// spec holds configuration that could apply to any operator.
	// +required
	Spec ClusterOperatorSpec `json:"spec"`

	// status holds the information about the state of an operator.  It is consistent with status information across
	// the Kubernetes ecosystem.
	// +optional
	Status ClusterOperatorStatus `json:"status"`
}

// ClusterOperatorSpec is empty for now, but you could imagine holding information like "pause".
type ClusterOperatorSpec struct {
}

// ClusterOperatorStatus provides information about the status of the operator.
// +k8s:deepcopy-gen=true
type ClusterOperatorStatus struct {
	// conditions describes the state of the operator's managed and monitored components.
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +optional
	Conditions []ClusterOperatorStatusCondition `json:"conditions,omitempty"  patchStrategy:"merge" patchMergeKey:"type"`

	// versions is a slice of operator and operand version tuples.  Operators which manage multiple operands will have multiple
	// operand entries in the array.  Available operators must report the version of the operator itself with the name "operator".
	// An operator reports a new "operator" version when it has rolled out the new version to all of its operands.
	// +optional
	Versions []OperandVersion `json:"versions,omitempty"`

	// relatedObjects is a list of objects that are "interesting" or related to this operator.  Common uses are:
	// 1. the detailed resource driving the operator
	// 2. operator namespaces
	// 3. operand namespaces
	// +optional
	RelatedObjects []ObjectReference `json:"relatedObjects,omitempty"`

	// extension contains any additional status information specific to the
	// operator which owns this status object.
	// +nullable
	// +optional
	// +kubebuilder:pruning:PreserveUnknownFields
	Extension runtime.RawExtension `json:"extension"`
}

type OperandVersion struct {
	// name is the name of the particular operand this version is for.  It usually matches container images, not operators.
	// +required
	Name string `json:"name"`

	// version indicates which version of a particular operand is currently being managed.  It must always match the Available
	// operand.  If 1.0.0 is Available, then this must indicate 1.0.0 even if the operator is trying to rollout
	// 1.1.0
	// +required
	Version string `json:"version"`
}

// ObjectReference contains enough information to let you inspect or modify the referred object.
type ObjectReference struct {
	// group of the referent.
	// +required
	Group string `json:"group"`
	// resource of the referent.
	// +required
	Resource string `json:"resource"`
	// namespace of the referent.
	// +optional
	Namespace string `json:"namespace,omitempty"`
	// name of the referent.
	// +required
	Name string `json:"name"`
}

type ConditionStatus string

// These are valid condition statuses. "ConditionTrue" means a resource is in the condition.
// "ConditionFalse" means a resource is not in the condition. "ConditionUnknown" means kubernetes
// can't decide if a resource is in the condition or not. In the future, we could add other
// intermediate conditions, e.g. ConditionDegraded.
const (
	ConditionTrue    ConditionStatus = "True"
	ConditionFalse   ConditionStatus = "False"
	ConditionUnknown ConditionStatus = "Unknown"
)

// ClusterOperatorStatusCondition represents the state of the operator's
// managed and monitored components.
// +k8s:deepcopy-gen=true
type ClusterOperatorStatusCondition struct {
	// type specifies the aspect reported by this condition.
	// +required
	Type ClusterStatusConditionType `json:"type"`

	// status of the condition, one of True, False, Unknown.
	// +required
	Status ConditionStatus `json:"status"`

	// lastTransitionTime is the time of the last update to the current status property.
	// +required
	LastTransitionTime metav1.Time `json:"lastTransitionTime"`

	// reason is the CamelCase reason for the condition's current status.
	// +optional
	Reason string `json:"reason,omitempty"`

	// message provides additional information about the current condition.
	// This is only to be consumed by humans.  It may contain Line Feed
	// characters (U+000A), which should be rendered as new lines.
	// +optional
	Message string `json:"message,omitempty"`
}

// ClusterStatusConditionType is an aspect of operator state.
type ClusterStatusConditionType string

const (
	// Available indicates that the component (operator and all configured operands)
	// is functional and available in the cluster. Available=False means at least
	// part of the component is non-functional, and that the condition requires
	// immediate administrator intervention.
	OperatorAvailable ClusterStatusConditionType = "Available"

	// Progressing indicates that the component (operator and all configured operands)
	// is actively rolling out new code, propagating config changes, or otherwise
	// moving from one steady state to another. Operators should not report
	// progressing when they are reconciling (without action) a previously known
	// state. If the observed cluster state has changed and the component is
	// reacting to it (scaling up for instance), Progressing should become true
	// since it is moving from one steady state to another.
	OperatorProgressing ClusterStatusConditionType = "Progressing"

	// Degraded indicates that the component (operator and all configured operands)
	// does not match its desired state over a period of time resulting in a lower
	// quality of service. The period of time may vary by component, but a Degraded
	// state represents persistent observation of a condition. As a result, a
	// component should not oscillate in and out of Degraded state. A component may
	// be Available even if its degraded. For example, a component may desire 3
	// running pods, but 1 pod is crash-looping. The component is Available but
	// Degraded because it may have a lower quality of service. A component may be
	// Progressing but not Degraded because the transition from one state to
	// another does not persist over a long enough period to report Degraded. A
	// component should not report Degraded during the course of a normal upgrade.
	// A component may report Degraded in response to a persistent infrastructure
	// failure that requires eventual administrator intervention.  For example, if
	// a control plane host is unhealthy and must be replaced. A component should
	// report Degraded if unexpected errors occur over a period, but the
	// expectation is that all unexpected errors are handled as operators mature.
	OperatorDegraded ClusterStatusConditionType = "Degraded"

	// Upgradeable indicates whether the component (operator and all configured
	// operands) is safe to upgrade based on the current cluster state. When
	// Upgradeable is False, the cluster-version operator will prevent the
	// cluster from performing impacted updates unless forced.  When set on
	// ClusterVersion, the message will explain which updates (minor or patch)
	// are impacted. When set on ClusterOperator, False will block minor
	// OpenShift updates. The message field should contain a human readable
	// description of what the administrator should do to allow the cluster or
	// component to successfully update. The cluster-version operator will
	// allow updates when this condition is not False, including when it is
	// missing, True, or Unknown.
	OperatorUpgradeable ClusterStatusConditionType = "Upgradeable"

	// EvaluationConditionsDetected is used to indicate the result of the detection
	// logic that was added to a component to evaluate the introduction of an
	// invasive change that could potentially result in highly visible alerts,
	// breakages or upgrade failures. You can concatenate multiple Reason using
	// the "::" delimiter if you need to evaluate the introduction of multiple changes.
	EvaluationConditionsDetected ClusterStatusConditionType = "EvaluationConditionsDetected"
)

// ClusterOperatorList is a list of OperatorStatus resources.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +openshift:compatibility-gen:level=1
type ClusterOperatorList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []ClusterOperator `json:"items"`
}
