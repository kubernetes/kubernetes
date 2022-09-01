package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// KubeControllerManager provides information to configure an operator to manage kube-controller-manager.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type KubeControllerManager struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata"`

	// spec is the specification of the desired behavior of the Kubernetes Controller Manager
	// +kubebuilder:validation:Required
	// +required
	Spec KubeControllerManagerSpec `json:"spec"`

	// status is the most recently observed status of the Kubernetes Controller Manager
	// +optional
	Status KubeControllerManagerStatus `json:"status"`
}

type KubeControllerManagerSpec struct {
	StaticPodOperatorSpec `json:",inline"`

	// useMoreSecureServiceCA indicates that the service-ca.crt provided in SA token volumes should include only
	// enough certificates to validate service serving certificates.
	// Once set to true, it cannot be set to false.
	// Even if someone finds a way to set it back to false, the service-ca.crt files that previously existed will
	// only have the more secure content.
	// +kubebuilder:default=false
	UseMoreSecureServiceCA bool `json:"useMoreSecureServiceCA"`
}

type KubeControllerManagerStatus struct {
	StaticPodOperatorStatus `json:",inline"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// KubeControllerManagerList is a collection of items
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type KubeControllerManagerList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	// Items contains the items
	Items []KubeControllerManager `json:"items"`
}
