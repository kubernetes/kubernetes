package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Storage provides a means to configure an operator to manage the cluster storage operator. `cluster` is the canonical name.
type Storage struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +kubebuilder:validation:Required
	// +required
	Spec StorageSpec `json:"spec"`

	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status StorageStatus `json:"status"`
}

// StorageSpec is the specification of the desired behavior of the cluster storage operator.
type StorageSpec struct {
	OperatorSpec `json:",inline"`
}

// StorageStatus defines the observed status of the cluster storage operator.
type StorageStatus struct {
	OperatorStatus `json:",inline"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:object:root=true

// StorageList contains a list of Storages.
type StorageList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Storage `json:"items"`
}
