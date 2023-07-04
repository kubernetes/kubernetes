package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Storage provides a means to configure an operator to manage the cluster storage operator. `cluster` is the canonical name.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type Storage struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +kubebuilder:validation:Required
	// +required
	Spec StorageSpec `json:"spec"`

	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status StorageStatus `json:"status"`
}

// StorageDriverType indicates whether CSI migration should be enabled for drivers where it is optional.
// +kubebuilder:validation:Enum="";LegacyDeprecatedInTreeDriver;CSIWithMigrationDriver
type StorageDriverType string

const (
	LegacyDeprecatedInTreeDriver StorageDriverType = "LegacyDeprecatedInTreeDriver"
	CSIWithMigrationDriver       StorageDriverType = "CSIWithMigrationDriver"
)

// StorageSpec is the specification of the desired behavior of the cluster storage operator.
type StorageSpec struct {
	OperatorSpec `json:",inline"`

	// VSphereStorageDriver indicates the storage driver to use on VSphere clusters.
	// Once this field is set to CSIWithMigrationDriver, it can not be changed.
	// If this is empty, the platform will choose a good default,
	// which may change over time without notice.
	// The current default is CSIWithMigrationDriver and may not be changed.
	// DEPRECATED: This field will be removed in a future release.
	// +kubebuilder:validation:XValidation:rule="self != \"LegacyDeprecatedInTreeDriver\"",message="VSphereStorageDriver can not be set to LegacyDeprecatedInTreeDriver"
	// +optional
	VSphereStorageDriver StorageDriverType `json:"vsphereStorageDriver"`
}

// StorageStatus defines the observed status of the cluster storage operator.
type StorageStatus struct {
	OperatorStatus `json:",inline"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:object:root=true

// StorageList contains a list of Storages.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type StorageList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty"`

	Items []Storage `json:"items"`
}
