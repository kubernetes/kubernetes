package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Config provides information to configure the config operator.
type Config struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata"`

	// spec is the specification of the desired behavior of the Config Operator.
	// +kubebuilder:validation:Required
	// +required
	Spec ConfigSpec `json:"spec"`

	// status defines the observed status of the Config Operator.
	// +optional
	Status ConfigStatus `json:"status"`
}

type ConfigSpec struct {
	OperatorSpec `json:",inline"`
}

type ConfigStatus struct {
	OperatorStatus `json:",inline"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ConfigList is a collection of items
type ConfigList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	// Items contains the items
	Items []Config `json:"items"`
}
