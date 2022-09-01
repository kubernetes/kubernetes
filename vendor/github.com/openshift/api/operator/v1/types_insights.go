package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
//
// InsightsOperator holds cluster-wide information about the Insights Operator.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type InsightsOperator struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata"`

	// spec is the specification of the desired behavior of the Insights.
	// +kubebuilder:validation:Required
	Spec InsightsOperatorSpec `json:"spec"`

	// status is the most recently observed status of the Insights operator.
	// +optional
	Status InsightsOperatorStatus `json:"status"`
}

type InsightsOperatorSpec struct {
	OperatorSpec `json:",inline"`
}

type InsightsOperatorStatus struct {
	OperatorStatus `json:",inline"`
	// gatherStatus provides basic information about the last Insights data gathering.
	// When omitted, this means no data gathering has taken place yet.
	// +optional
	GatherStatus GatherStatus `json:"gatherStatus,omitempty"`
	// insightsReport provides general Insights analysis results.
	// When omitted, this means no data gathering has taken place yet.
	// +optional
	InsightsReport InsightsReport `json:"insightsReport,omitempty"`
}

// gatherStatus provides information about the last known gather event.
type GatherStatus struct {
	// lastGatherTime is the last time when Insights data gathering finished.
	// An empty value means that no data has been gathered yet.
	// +optional
	LastGatherTime metav1.Time `json:"lastGatherTime,omitempty"`
	// lastGatherDuration is the total time taken to process
	// all gatherers during the last gather event.
	// +optional
	// +kubebuilder:validation:Pattern="^0|([1-9][0-9]*(\\.[0-9]+)?(ns|us|µs|ms|s|m|h))+$"
	// +kubebuilder:validation:Type=string
	LastGatherDuration metav1.Duration `json:"lastGatherDuration,omitempty"`
	// gatherers is a list of active gatherers (and their statuses) in the last gathering.
	// +listType=atomic
	// +optional
	Gatherers []GathererStatus `json:"gatherers,omitempty"`
}

// insightsReport provides Insights health check report based on the most
// recently sent Insights data.
type InsightsReport struct {
	// healthChecks provides basic information about active Insights health checks
	// in a cluster.
	// +listType=atomic
	// +optional
	HealthChecks []HealthCheck `json:"healthChecks,omitempty"`
}

// healthCheck represents an Insights health check attributes.
type HealthCheck struct {
	// description provides basic description of the healtcheck.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxLength=2048
	// +kubebuilder:validation:MinLength=10
	Description string `json:"description"`
	// totalRisk of the healthcheck. Indicator of the total risk posed
	// by the detected issue; combination of impact and likelihood. The values can be from 1 to 4,
	// and the higher the number, the more important the issue.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=4
	TotalRisk int32 `json:"totalRisk"`
	// advisorURI provides the URL link to the Insights Advisor.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Pattern=`^https:\/\/\S+`
	AdvisorURI string `json:"advisorURI"`
	// state determines what the current state of the health check is.
	// Health check is enabled by default and can be disabled
	// by the user in the Insights advisor user interface.
	// +kubebuilder:validation:Required
	State HealthCheckState `json:"state"`
}

// healthCheckState provides information about the status of the
// health check (for example, the health check may be marked as disabled by the user).
// +kubebuilder:validation:Enum:=Enabled;Disabled
type HealthCheckState string

const (
	// enabled marks the health check as enabled
	HealthCheckEnabled HealthCheckState = "Enabled"
	// disabled marks the health check as disabled
	HealthCheckDisabled HealthCheckState = "Disabled"
)

// gathererStatus represents information about a particular
// data gatherer.
type GathererStatus struct {
	// conditions provide details on the status of each gatherer.
	// +listType=atomic
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinItems=1
	Conditions []metav1.Condition `json:"conditions"`
	// name is the name of the gatherer.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxLength=256
	// +kubebuilder:validation:MinLength=5
	Name string `json:"name"`
	// lastGatherDuration represents the time spent gathering.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Type=string
	// +kubebuilder:validation:Pattern="^([1-9][0-9]*(\\.[0-9]+)?(ns|us|µs|ms|s|m|h))+$"
	LastGatherDuration metav1.Duration `json:"lastGatherDuration"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// InsightsOperatorList is a collection of items
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type InsightsOperatorList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`
	Items           []InsightsOperator `json:"items"`
}
