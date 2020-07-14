package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// OperatorHubSpec defines the desired state of OperatorHub
type OperatorHubSpec struct {
	// disableAllDefaultSources allows you to disable all the default hub
	// sources. If this is true, a specific entry in sources can be used to
	// enable a default source. If this is false, a specific entry in
	// sources can be used to disable or enable a default source.
	// +optional
	DisableAllDefaultSources bool `json:"disableAllDefaultSources,omitempty"`
	// sources is the list of default hub sources and their configuration.
	// If the list is empty, it implies that the default hub sources are
	// enabled on the cluster unless disableAllDefaultSources is true.
	// If disableAllDefaultSources is true and sources is not empty,
	// the configuration present in sources will take precedence. The list of
	// default hub sources and their current state will always be reflected in
	// the status block.
	// +optional
	Sources []HubSource `json:"sources,omitempty"`
}

// OperatorHubStatus defines the observed state of OperatorHub. The current
// state of the default hub sources will always be reflected here.
type OperatorHubStatus struct {
	// sources encapsulates the result of applying the configuration for each
	// hub source
	Sources []HubSourceStatus `json:"sources,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// OperatorHub is the Schema for the operatorhubs API. It can be used to change
// the state of the default hub sources for OperatorHub on the cluster from
// enabled to disabled and vice versa.
// +kubebuilder:subresource:status
// +genclient
// +genclient:nonNamespaced
type OperatorHub struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata"`

	Spec   OperatorHubSpec   `json:"spec"`
	Status OperatorHubStatus `json:"status"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// OperatorHubList contains a list of OperatorHub
type OperatorHubList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`
	Items           []OperatorHub `json:"items"`
}

// HubSource is used to specify the hub source and its configuration
type HubSource struct {
	// name is the name of one of the default hub sources
	// +kubebuilder:validation:MaxLength=253
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:Required
	Name string `json:"name"`
	// disabled is used to disable a default hub source on cluster
	// +kubebuilder:Required
	Disabled bool `json:"disabled"`
}

// HubSourceStatus is used to reflect the current state of applying the
// configuration to a default source
type HubSourceStatus struct {
	HubSource `json:",omitempty"`
	// status indicates success or failure in applying the configuration
	Status string `json:"status,omitempty"`
	// message provides more information regarding failures
	Message string `json:"message,omitempty"`
}
