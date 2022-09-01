package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Feature holds cluster-wide information about feature gates.  The canonical name is `cluster`
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type FeatureGate struct {
	metav1.TypeMeta   `json:",inline"`
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

	// CustomNoUpgrade allows the enabling or disabling of any feature. Turning this feature set on IS NOT SUPPORTED, CANNOT BE UNDONE, and PREVENTS UPGRADES.
	// Because of its nature, this setting cannot be validated.  If you have any typos or accidentally apply invalid combinations
	// your cluster may fail in an unrecoverable way.
	CustomNoUpgrade FeatureSet = "CustomNoUpgrade"

	// TopologyManager enables ToplogyManager support. Upgrades are enabled with this feature.
	LatencySensitive FeatureSet = "LatencySensitive"
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
	Enabled []string `json:"enabled,omitempty"`
	// disabled is a list of all feature gates that you want to force off
	// +optional
	Disabled []string `json:"disabled,omitempty"`
}

type FeatureGateStatus struct {
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type FeatureGateList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []FeatureGate `json:"items"`
}

type FeatureGateEnabledDisabled struct {
	Enabled  []string
	Disabled []string
}

// FeatureSets Contains a map of Feature names to Enabled/Disabled Feature.
//
// NOTE: The caller needs to make sure to check for the existence of the value
// using golang's existence field. A possible scenario is an upgrade where new
// FeatureSets are added and a controller has not been upgraded with a newer
// version of this file. In this upgrade scenario the map could return nil.
//
// example:
//
//	if featureSet, ok := FeatureSets["SomeNewFeature"]; ok { }
//
// If you put an item in either of these lists, put your area and name on it so we can find owners.
var FeatureSets = map[FeatureSet]*FeatureGateEnabledDisabled{
	Default: defaultFeatures,
	CustomNoUpgrade: {
		Enabled:  []string{},
		Disabled: []string{},
	},
	TechPreviewNoUpgrade: newDefaultFeatures().
		with("CSIMigrationAzureFile").       // sig-storage, fbertina, Kubernetes feature gate
		with("CSIMigrationvSphere").         // sig-storage, fbertina, Kubernetes feature gate
		with("ExternalCloudProvider").       // sig-cloud-provider, jspeed, OCP specific
		with("CSIDriverSharedResource").     // sig-build, adkaplan, OCP specific
		with("BuildCSIVolumes").             // sig-build, adkaplan, OCP specific
		with("NodeSwap").                    // sig-node, ehashman, Kubernetes feature gate
		with("MachineAPIProviderOpenStack"). // openstack, egarcia (#forum-openstack), OCP specific
		with("CGroupsV2").                   // sig-node, harche, OCP specific
		with("Crun").                        // sig-node, haircommander, OCP specific
		toFeatures(),
	LatencySensitive: newDefaultFeatures().
		with(
			"TopologyManager", // sig-pod, sjenning
		).
		toFeatures(),
}

var defaultFeatures = &FeatureGateEnabledDisabled{
	Enabled: []string{
		"APIPriorityAndFairness",         // sig-apimachinery, deads2k
		"RotateKubeletServerCertificate", // sig-pod, sjenning
		"DownwardAPIHugePages",           // sig-node, rphillips
	},
	Disabled: []string{
		"CSIMigrationAzureFile", // sig-storage, jsafrane
		"CSIMigrationvSphere",   // sig-storage, jsafrane
	},
}

type featureSetBuilder struct {
	forceOn  []string
	forceOff []string
}

func newDefaultFeatures() *featureSetBuilder {
	return &featureSetBuilder{}
}

func (f *featureSetBuilder) with(forceOn ...string) *featureSetBuilder {
	f.forceOn = append(f.forceOn, forceOn...)
	return f
}

func (f *featureSetBuilder) without(forceOff ...string) *featureSetBuilder {
	f.forceOff = append(f.forceOff, forceOff...)
	return f
}

func (f *featureSetBuilder) isForcedOff(needle string) bool {
	for _, forcedOff := range f.forceOff {
		if needle == forcedOff {
			return true
		}
	}
	return false
}

func (f *featureSetBuilder) isForcedOn(needle string) bool {
	for _, forceOn := range f.forceOn {
		if needle == forceOn {
			return true
		}
	}
	return false
}

func (f *featureSetBuilder) toFeatures() *FeatureGateEnabledDisabled {
	finalOn := []string{}
	finalOff := []string{}

	// only add the default enabled features if they haven't been explicitly set off
	for _, defaultOn := range defaultFeatures.Enabled {
		if !f.isForcedOff(defaultOn) {
			finalOn = append(finalOn, defaultOn)
		}
	}
	for _, currOn := range f.forceOn {
		if f.isForcedOff(currOn) {
			panic("coding error, you can't have features both on and off")
		}
		finalOn = append(finalOn, currOn)
	}

	// only add the default disabled features if they haven't been explicitly set on
	for _, defaultOff := range defaultFeatures.Disabled {
		if !f.isForcedOn(defaultOff) {
			finalOff = append(finalOff, defaultOff)
		}
	}
	for _, currOff := range f.forceOff {
		finalOff = append(finalOff, currOff)
	}

	return &FeatureGateEnabledDisabled{
		Enabled:  finalOn,
		Disabled: finalOff,
	}
}
