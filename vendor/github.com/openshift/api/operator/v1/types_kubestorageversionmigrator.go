package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// KubeStorageVersionMigrator provides information to configure an operator to manage kube-storage-version-migrator.
type KubeStorageVersionMigrator struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata"`

	// +kubebuilder:validation:Required
	// +required
	Spec KubeStorageVersionMigratorSpec `json:"spec"`
	// +optional
	Status KubeStorageVersionMigratorStatus `json:"status"`
}

type KubeStorageVersionMigratorSpec struct {
	OperatorSpec `json:",inline"`
}

type KubeStorageVersionMigratorStatus struct {
	OperatorStatus `json:",inline"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// KubeStorageVersionMigratorList is a collection of items
type KubeStorageVersionMigratorList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	// Items contains the items
	Items []KubeStorageVersionMigrator `json:"items"`
}
