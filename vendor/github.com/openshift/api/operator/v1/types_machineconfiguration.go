package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// MachineConfiguration provides information to configure an operator to manage Machine Configuration.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type MachineConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata"`

	// spec is the specification of the desired behavior of the Machine Config Operator
	// +kubebuilder:validation:Required
	Spec MachineConfigurationSpec `json:"spec"`

	// status is the most recently observed status of the Machine Config Operator
	// +optional
	Status MachineConfigurationStatus `json:"status"`
}

type MachineConfigurationSpec struct {
	StaticPodOperatorSpec `json:",inline"`

	// TODO(jkyros): This is where we put our knobs and dials
}

type MachineConfigurationStatus struct {
	StaticPodOperatorStatus `json:",inline"`

	// TODO(jkyros): This is where we can put additional bespoke status fields
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// MachineConfigurationList is a collection of items
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type MachineConfigurationList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	// Items contains the items
	Items []MachineConfiguration `json:"items"`
}
