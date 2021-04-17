package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// OpenShiftControllerManager provides information to configure an operator to manage openshift-controller-manager.
type OpenShiftControllerManager struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata"`

	// +kubebuilder:validation:Required
	// +required
	Spec OpenShiftControllerManagerSpec `json:"spec"`
	// +optional
	Status OpenShiftControllerManagerStatus `json:"status"`
}

type OpenShiftControllerManagerSpec struct {
	OperatorSpec `json:",inline"`
}

type OpenShiftControllerManagerStatus struct {
	OperatorStatus `json:",inline"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// OpenShiftControllerManagerList is a collection of items
type OpenShiftControllerManagerList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	// Items contains the items
	Items []OpenShiftControllerManager `json:"items"`
}
