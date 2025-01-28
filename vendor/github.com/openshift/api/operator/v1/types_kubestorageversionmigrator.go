package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=kubestorageversionmigrators,scope=Cluster
// +kubebuilder:subresource:status
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/503
// +openshift:file-pattern=cvoRunLevel=0000_40,operatorName=kube-storage-version-migrator,operatorOrdering=00

// KubeStorageVersionMigrator provides information to configure an operator to manage kube-storage-version-migrator.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type KubeStorageVersionMigrator struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata"`

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
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type KubeStorageVersionMigratorList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	// items contains the items
	Items []KubeStorageVersionMigrator `json:"items"`
}
