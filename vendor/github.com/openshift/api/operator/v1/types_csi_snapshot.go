package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// CSISnapshotController provides a means to configure an operator to manage the CSI snapshots. `cluster` is the canonical name.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type CSISnapshotController struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +kubebuilder:validation:Required
	// +required
	Spec CSISnapshotControllerSpec `json:"spec"`

	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status CSISnapshotControllerStatus `json:"status"`
}

// CSISnapshotControllerSpec is the specification of the desired behavior of the CSISnapshotController operator.
type CSISnapshotControllerSpec struct {
	OperatorSpec `json:",inline"`
}

// CSISnapshotControllerStatus defines the observed status of the CSISnapshotController operator.
type CSISnapshotControllerStatus struct {
	OperatorStatus `json:",inline"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:object:root=true

// CSISnapshotControllerList contains a list of CSISnapshotControllers.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type CSISnapshotControllerList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []CSISnapshotController `json:"items"`
}
