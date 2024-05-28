package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Feature holds cluster-wide information about feature gates.  The canonical name is `cluster`
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/470
// +openshift:file-pattern=cvoRunLevel=0000_10,operatorName=config-operator,operatorOrdering=01
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=featuregates,scope=Cluster
// +kubebuilder:subresource:status
// +kubebuilder:metadata:annotations=release.openshift.io/bootstrap-required=true
type FeatureGate struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +kubebuilder:validation:Required
	// +required
	Spec FeatureGateSpec `json:"spec"`
	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status FeatureGateStatus `json:"status"`
}

type FeatureSet string

var (
	// Default feature set that allows upgrades.
	Default FeatureSet = ""

	// TechPreviewNoUpgrade turns on tech preview features that are not part of the normal supported platform. Turning
	// this feature set on CANNOT BE UNDONE and PREVENTS UPGRADES.
	TechPreviewNoUpgrade FeatureSet = "TechPreviewNoUpgrade"

	// DevPreviewNoUpgrade turns on dev preview features that are not part of the normal supported platform. Turning
	// this feature set on CANNOT BE UNDONE and PREVENTS UPGRADES.
	DevPreviewNoUpgrade FeatureSet = "DevPreviewNoUpgrade"

	// CustomNoUpgrade allows the enabling or disabling of any feature. Turning this feature set on IS NOT SUPPORTED, CANNOT BE UNDONE, and PREVENTS UPGRADES.
	// Because of its nature, this setting cannot be validated.  If you have any typos or accidentally apply invalid combinations
	// your cluster may fail in an unrecoverable way.
	CustomNoUpgrade FeatureSet = "CustomNoUpgrade"

	// AllFixedFeatureSets are the featuresets that have known featuregates.  Custom doesn't for instance.  LatencySensitive is dead
	AllFixedFeatureSets = []FeatureSet{Default, TechPreviewNoUpgrade, DevPreviewNoUpgrade}
)

type FeatureGateSpec struct {
	FeatureGateSelection `json:",inline"`
}

// +union
type FeatureGateSelection struct {
	// featureSet changes the list of features in the cluster.  The default is empty.  Be very careful adjusting this setting.
	// Turning on or off features may cause irreversible changes in your cluster which cannot be undone.
	// +unionDiscriminator
	// +optional
	// +kubebuilder:validation:XValidation:rule="oldSelf == 'CustomNoUpgrade' ? self == 'CustomNoUpgrade' : true",message="CustomNoUpgrade may not be changed"
	// +kubebuilder:validation:XValidation:rule="oldSelf == 'TechPreviewNoUpgrade' ? self == 'TechPreviewNoUpgrade' : true",message="TechPreviewNoUpgrade may not be changed"
	// +kubebuilder:validation:XValidation:rule="oldSelf == 'DevPreviewNoUpgrade' ? self == 'DevPreviewNoUpgrade' : true",message="DevPreviewNoUpgrade may not be changed"
	FeatureSet FeatureSet `json:"featureSet,omitempty"`

	// customNoUpgrade allows the enabling or disabling of any feature. Turning this feature set on IS NOT SUPPORTED, CANNOT BE UNDONE, and PREVENTS UPGRADES.
	// Because of its nature, this setting cannot be validated.  If you have any typos or accidentally apply invalid combinations
	// your cluster may fail in an unrecoverable way.  featureSet must equal "CustomNoUpgrade" must be set to use this field.
	// +optional
	// +nullable
	CustomNoUpgrade *CustomFeatureGates `json:"customNoUpgrade,omitempty"`
}

type CustomFeatureGates struct {
	// enabled is a list of all feature gates that you want to force on
	// +optional
	Enabled []FeatureGateName `json:"enabled,omitempty"`
	// disabled is a list of all feature gates that you want to force off
	// +optional
	Disabled []FeatureGateName `json:"disabled,omitempty"`
}

// FeatureGateName is a string to enforce patterns on the name of a FeatureGate
// +kubebuilder:validation:Pattern=`^([A-Za-z0-9-]+\.)*[A-Za-z0-9-]+\.?$`
type FeatureGateName string

type FeatureGateStatus struct {
	// conditions represent the observations of the current state.
	// Known .status.conditions.type are: "DeterminationDegraded"
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// featureGates contains a list of enabled and disabled featureGates that are keyed by payloadVersion.
	// Operators other than the CVO and cluster-config-operator, must read the .status.featureGates, locate
	// the version they are managing, find the enabled/disabled featuregates and make the operand and operator match.
	// The enabled/disabled values for a particular version may change during the life of the cluster as various
	// .spec.featureSet values are selected.
	// Operators may choose to restart their processes to pick up these changes, but remembering past enable/disable
	// lists is beyond the scope of this API and is the responsibility of individual operators.
	// Only featureGates with .version in the ClusterVersion.status will be present in this list.
	// +listType=map
	// +listMapKey=version
	FeatureGates []FeatureGateDetails `json:"featureGates"`
}

type FeatureGateDetails struct {
	// version matches the version provided by the ClusterVersion and in the ClusterOperator.Status.Versions field.
	// +kubebuilder:validation:Required
	// +required
	Version string `json:"version"`
	// enabled is a list of all feature gates that are enabled in the cluster for the named version.
	// +optional
	Enabled []FeatureGateAttributes `json:"enabled"`
	// disabled is a list of all feature gates that are disabled in the cluster for the named version.
	// +optional
	Disabled []FeatureGateAttributes `json:"disabled"`
}

type FeatureGateAttributes struct {
	// name is the name of the FeatureGate.
	// +kubebuilder:validation:Required
	Name FeatureGateName `json:"name"`

	// possible (probable?) future additions include
	// 1. support level (Stable, ServiceDeliveryOnly, TechPreview, DevPreview)
	// 2. description
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type FeatureGateList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []FeatureGate `json:"items"`
}
