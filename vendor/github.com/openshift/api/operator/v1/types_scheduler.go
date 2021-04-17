package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// KubeScheduler provides information to configure an operator to manage scheduler.
type KubeScheduler struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata"`

	// spec is the specification of the desired behavior of the Kubernetes Scheduler
	// +kubebuilder:validation:Required
	// +required
	Spec KubeSchedulerSpec `json:"spec"`

	// status is the most recently observed status of the Kubernetes Scheduler
	// +optional
	Status KubeSchedulerStatus `json:"status"`
}

type KubeSchedulerSpec struct {
	StaticPodOperatorSpec `json:",inline"`
}

type KubeSchedulerStatus struct {
	StaticPodOperatorStatus `json:",inline"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// KubeSchedulerList is a collection of items
type KubeSchedulerList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	// Items contains the items
	Items []KubeScheduler `json:"items"`
}
