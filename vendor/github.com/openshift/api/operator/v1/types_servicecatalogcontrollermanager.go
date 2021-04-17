package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ServiceCatalogControllerManager provides information to configure an operator to manage Service Catalog Controller Manager
// DEPRECATED: will be removed in 4.6
type ServiceCatalogControllerManager struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata"`

	// +kubebuilder:validation:Required
	// +required
	Spec ServiceCatalogControllerManagerSpec `json:"spec"`
	// +optional
	Status ServiceCatalogControllerManagerStatus `json:"status"`
}

type ServiceCatalogControllerManagerSpec struct {
	OperatorSpec `json:",inline"`
}

type ServiceCatalogControllerManagerStatus struct {
	OperatorStatus `json:",inline"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ServiceCatalogControllerManagerList is a collection of items
// DEPRECATED: will be removed in 4.6
type ServiceCatalogControllerManagerList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	// Items contains the items
	Items []ServiceCatalogControllerManager `json:"items"`
}
