package v1alpha2

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
//
// InsightsDataGather provides data gather configuration options for the the Insights Operator.
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=insightsdatagathers,scope=Cluster
// +kubebuilder:subresource:status
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/2195
// +openshift:file-pattern=cvoRunLevel=0000_10,operatorName=config-operator,operatorOrdering=01
// +openshift:enable:FeatureGate=InsightsConfig
// +openshift:compatibility-gen:level=4
// +openshift:capability=Insights
type InsightsDataGather struct {
	metav1.TypeMeta `json:",inline"`
	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`
	// spec holds user settable values for configuration
	// +required
	Spec InsightsDataGatherSpec `json:"spec"`
	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status InsightsDataGatherStatus `json:"status"`
}

type InsightsDataGatherSpec struct {
	// gatherConfig is an optional spec attribute that includes all the configuration options related to gathering of the Insights data and its uploading to the ingress.
	// +optional
	GatherConfig GatherConfig `json:"gatherConfig"`
}

type InsightsDataGatherStatus struct{}

// gatherConfig provides data gathering configuration options.
type GatherConfig struct {
	// dataPolicy is an optional list of DataPolicyOptions that allows user to enable additional obfuscation of the Insights archive data.
	// It may not exceed 2 items and must not contain duplicates.
	// Valid values are ObfuscateNetworking and WorkloadNames.
	// When set to ObfuscateNetworking the IP addresses and the cluster domain name are obfuscated.
	// When set to WorkloadNames, the gathered data about cluster resources will not contain the workload names for your deployments. Resources UIDs will be used instead.
	// When omitted no obfuscation is applied.
	// +kubebuilder:validation:MaxItems=2
	// +kubebuilder:validation:XValidation:rule="self.all(x, self.exists_one(y, x == y))",message="dataPolicy items must be unique"
	// +listType=atomic
	// +optional
	DataPolicy []DataPolicyOption `json:"dataPolicy"`
	// gatherers is a required field that specifies the configuration of the gatherers.
	// +required
	Gatherers Gatherers `json:"gatherers"`
	// storage is an optional field that allows user to define persistent storage for gathering jobs to store the Insights data archive.
	// If omitted, the gathering job will use ephemeral storage.
	// +optional
	Storage *Storage `json:"storage,omitempty"`
}

// +kubebuilder:validation:XValidation:rule="has(self.mode) && self.mode == 'Custom' ?  has(self.custom) : !has(self.custom)",message="custom is required when mode is Custom, and forbidden otherwise"
type Gatherers struct {
	// mode is a required field that specifies the mode for gatherers. Allowed values are All, None, and Custom.
	// When set to All, all gatherers wil run and gather data.
	// When set to None, all gatherers will be disabled and no data will be gathered.
	// When set to Custom, the custom configuration from the custom field will be applied.
	// +required
	Mode GatheringMode `json:"mode"`
	// custom provides gathering configuration.
	// It is required when mode is Custom, and forbidden otherwise.
	// Custom configuration allows user to disable only a subset of gatherers.
	// Gatherers that are not explicitly disabled in custom configuration will run.
	// +optional
	Custom *Custom `json:"custom,omitempty"`
}

// custom provides the custom configuration of gatherers
type Custom struct {
	// configs is a required list of gatherers configurations that can be used to enable or disable specific gatherers.
	// It may not exceed 100 items and each gatherer can be present only once.
	// It is possible to disable an entire set of gatherers while allowing a specific function within that set.
	// The particular gatherers IDs can be found at https://github.com/openshift/insights-operator/blob/master/docs/gathered-data.md.
	// Run the following command to get the names of last active gatherers:
	// "oc get insightsoperators.operator.openshift.io cluster -o json | jq '.status.gatherStatus.gatherers[].name'"
	// +kubebuilder:validation:MaxItems=100
	// +listType=map
	// +listMapKey=name
	// +required
	Configs []GathererConfig `json:"configs"`
}

// gatheringMode defines the valid gathering modes.
// +kubebuilder:validation:Enum=All;None;Custom
type GatheringMode string

const (
	// Enabled enables all gatherers
	GatheringModeAll GatheringMode = "All"
	// Disabled disables all gatherers
	GatheringModeNone GatheringMode = "None"
	// Custom applies the configuration from GatheringConfig.
	GatheringModeCustom GatheringMode = "Custom"
)

// dataPolicyOption declares valid data policy options
// +kubebuilder:validation:Enum=ObfuscateNetworking;WorkloadNames
type DataPolicyOption string

const (
	// IP addresses and cluster domain name are obfuscated
	DataPolicyOptionObfuscateNetworking DataPolicyOption = "ObfuscateNetworking"
	// Data from Deployment Validation Operator are obfuscated
	DataPolicyOptionObfuscateWorkloadNames DataPolicyOption = "WorkloadNames"
)

// storage provides persistent storage configuration options for gathering jobs.
// If the type is set to PersistentVolume, then the PersistentVolume must be defined.
// If the type is set to Ephemeral, then the PersistentVolume must not be defined.
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'PersistentVolume' ?  has(self.persistentVolume) : !has(self.persistentVolume)",message="persistentVolume is required when type is PersistentVolume, and forbidden otherwise"
type Storage struct {
	// type is a required field that specifies the type of storage that will be used to store the Insights data archive.
	// Valid values are "PersistentVolume" and "Ephemeral".
	// When set to Ephemeral, the Insights data archive is stored in the ephemeral storage of the gathering job.
	// When set to PersistentVolume, the Insights data archive is stored in the PersistentVolume that is defined by the persistentVolume field.
	// +required
	Type StorageType `json:"type"`
	// persistentVolume is an optional field that specifies the PersistentVolume that will be used to store the Insights data archive.
	// The PersistentVolume must be created in the openshift-insights namespace.
	// +optional
	PersistentVolume *PersistentVolumeConfig `json:"persistentVolume,omitempty"`
}

// storageType declares valid storage types
// +kubebuilder:validation:Enum=PersistentVolume;Ephemeral
type StorageType string

const (
	// StorageTypePersistentVolume storage type
	StorageTypePersistentVolume StorageType = "PersistentVolume"
	// StorageTypeEphemeral storage type
	StorageTypeEphemeral StorageType = "Ephemeral"
)

// persistentVolumeConfig provides configuration options for PersistentVolume storage.
type PersistentVolumeConfig struct {
	// claim is a required field that specifies the configuration of the PersistentVolumeClaim that will be used to store the Insights data archive.
	// The PersistentVolumeClaim must be created in the openshift-insights namespace.
	// +required
	Claim PersistentVolumeClaimReference `json:"claim"`
	// mountPath is an optional field specifying the directory where the PVC will be mounted inside the Insights data gathering Pod.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// The current default mount path is /var/lib/insights-operator
	// The path may not exceed 1024 characters and must not contain a colon.
	// +kubebuilder:validation:MaxLength=1024
	// +kubebuilder:validation:XValidation:rule="!self.contains(':')",message="mountPath must not contain a colon"
	// +optional
	MountPath string `json:"mountPath,omitempty"`
}

// persistentVolumeClaimReference is a reference to a PersistentVolumeClaim.
type PersistentVolumeClaimReference struct {
	// name is a string that follows the DNS1123 subdomain format.
	// It must be at most 253 characters in length, and must consist only of lower case alphanumeric characters, '-' and '.', and must start and end with an alphanumeric character.
	// +kubebuilder:validation:XValidation:rule="!format.dns1123Subdomain().validate(self).hasValue()",message="a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character."
	// +kubebuilder:validation:MaxLength:=253
	// +required
	Name string `json:"name"`
}

// gathererConfig allows to configure specific gatherers
type GathererConfig struct {
	// name is the required name of a specific gatherer
	// It may not exceed 256 characters.
	// The format for a gatherer name is: {gatherer}/{function} where the function is optional.
	// Gatherer consists of a lowercase letters only that may include underscores (_).
	// Function consists of a lowercase letters only that may include underscores (_) and is separated from the gatherer by a forward slash (/).
	// The particular gatherers can be found at https://github.com/openshift/insights-operator/blob/master/docs/gathered-data.md.
	// Run the following command to get the names of last active gatherers:
	// "oc get insightsoperators.operator.openshift.io cluster -o json | jq '.status.gatherStatus.gatherers[].name'"
	// +kubebuilder:validation:MaxLength=256
	// +kubebuilder:validation:XValidation:rule=`self.matches("^[a-z]+[_a-z]*[a-z]([/a-z][_a-z]*)?[a-z]$")`,message=`gatherer name must be in the format of {gatherer}/{function} where the gatherer and function are lowercase letters only that may include underscores (_) and are separated by a forward slash (/) if the function is provided`
	// +required
	Name string `json:"name"`
	// state is a required field that allows you to configure specific gatherer. Valid values are "Enabled" and "Disabled".
	// When set to Enabled the gatherer will run.
	// When set to Disabled the gatherer will not run.
	// +required
	State GathererState `json:"state"`
}

// state declares valid gatherer state types.
// +kubebuilder:validation:Enum=Enabled;Disabled
type GathererState string

const (
	// GathererStateEnabled gatherer state, which means that the gatherer will run.
	GathererStateEnabled GathererState = "Enabled"
	// GathererStateDisabled gatherer state, which means that the gatherer will not run.
	GathererStateDisabled GathererState = "Disabled"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// InsightsDataGatherList is a collection of items
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
type InsightsDataGatherList struct {
	metav1.TypeMeta `json:",inline"`
	// metadata is the required standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +required
	metav1.ListMeta `json:"metadata"`
	// items is the required list of InsightsDataGather objects
	// it may not exceed 100 items
	// +kubebuilder:validation:MaxItems=100
	// +required
	Items []InsightsDataGather `json:"items"`
}
