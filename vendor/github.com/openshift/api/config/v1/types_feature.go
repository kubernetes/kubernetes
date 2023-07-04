package v1

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Feature holds cluster-wide information about feature gates.  The canonical name is `cluster`
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
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

type FeatureGateEnabledDisabled struct {
	Enabled  []FeatureGateDescription
	Disabled []FeatureGateDescription
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
		Enabled:  []FeatureGateDescription{},
		Disabled: []FeatureGateDescription{},
	},
	TechPreviewNoUpgrade: newDefaultFeatures().
		with(externalCloudProvider).
		with(externalCloudProviderAzure).
		with(externalCloudProviderGCP).
		with(externalCloudProviderExternal).
		with(csiDriverSharedResource).
		with(buildCSIVolumes).
		with(nodeSwap).
		with(machineAPIProviderOpenStack).
		with(insightsConfigAPI).
		with(retroactiveDefaultStorageClass).
		with(pdbUnhealthyPodEvictionPolicy).
		with(dynamicResourceAllocation).
		with(admissionWebhookMatchConditions).
		with(awsSecurityTokenService).
		with(azureWorkloadIdentity).
		with(gateGatewayAPI).
		with(maxUnavailableStatefulSet).
		without(eventedPleg).
		with(privateHostedZoneAWS).
		with(sigstoreImageVerification).
		with(gcpLabelsTags).
		with(vSphereStaticIPs).
		toFeatures(defaultFeatures),
	LatencySensitive: newDefaultFeatures().
		toFeatures(defaultFeatures),
}

var defaultFeatures = &FeatureGateEnabledDisabled{
	Enabled: []FeatureGateDescription{
		openShiftPodSecurityAdmission,
		alibabaPlatform, // This is a bug, it should be TechPreviewNoUpgrade. This must be downgraded before 4.14 is shipped.
		cloudDualStackNodeIPs,
	},
	Disabled: []FeatureGateDescription{
		retroactiveDefaultStorageClass,
	},
}

type featureSetBuilder struct {
	forceOn  []FeatureGateDescription
	forceOff []FeatureGateDescription
}

func newDefaultFeatures() *featureSetBuilder {
	return &featureSetBuilder{}
}

func (f *featureSetBuilder) with(forceOn FeatureGateDescription) *featureSetBuilder {
	for _, curr := range f.forceOn {
		if curr.FeatureGateAttributes.Name == forceOn.FeatureGateAttributes.Name {
			panic(fmt.Errorf("coding error: %q enabled twice", forceOn.FeatureGateAttributes.Name))
		}
	}
	f.forceOn = append(f.forceOn, forceOn)
	return f
}

func (f *featureSetBuilder) without(forceOff FeatureGateDescription) *featureSetBuilder {
	for _, curr := range f.forceOff {
		if curr.FeatureGateAttributes.Name == forceOff.FeatureGateAttributes.Name {
			panic(fmt.Errorf("coding error: %q disabled twice", forceOff.FeatureGateAttributes.Name))
		}
	}
	f.forceOff = append(f.forceOff, forceOff)
	return f
}

func (f *featureSetBuilder) isForcedOff(needle FeatureGateDescription) bool {
	for _, forcedOff := range f.forceOff {
		if needle.FeatureGateAttributes.Name == forcedOff.FeatureGateAttributes.Name {
			return true
		}
	}
	return false
}

func (f *featureSetBuilder) isForcedOn(needle FeatureGateDescription) bool {
	for _, forceOn := range f.forceOn {
		if needle.FeatureGateAttributes.Name == forceOn.FeatureGateAttributes.Name {
			return true
		}
	}
	return false
}

func (f *featureSetBuilder) toFeatures(defaultFeatures *FeatureGateEnabledDisabled) *FeatureGateEnabledDisabled {
	finalOn := []FeatureGateDescription{}
	finalOff := []FeatureGateDescription{}

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
		found := false
		for _, alreadyOn := range finalOn {
			if alreadyOn.FeatureGateAttributes.Name == currOn.FeatureGateAttributes.Name {
				found = true
			}
		}
		if found {
			continue
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
