package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=machineconfigurations,scope=Cluster
// +kubebuilder:subresource:status
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/1453
// +openshift:file-pattern=cvoRunLevel=0000_80,operatorName=machine-config,operatorOrdering=01

// MachineConfiguration provides information to configure an operator to manage Machine Configuration.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
// +openshift:validation:FeatureGateAwareXValidation:featureGate=BootImageSkewEnforcement,rule="self.?status.bootImageSkewEnforcementStatus.mode.orValue(\"\") == 'Automatic' ? self.?spec.managedBootImages.hasValue() || self.?status.managedBootImagesStatus.hasValue() : true",message="when skew enforcement is in Automatic mode, a boot image configuration is required"
// +openshift:validation:FeatureGateAwareXValidation:featureGate=BootImageSkewEnforcement,rule="self.?status.bootImageSkewEnforcementStatus.mode.orValue(\"\") == 'Automatic' ? !(self.?spec.managedBootImages.machineManagers.hasValue()) || size(self.spec.managedBootImages.machineManagers) > 0 : true",message="when skew enforcement is in Automatic mode, managedBootImages.machineManagers must not be an empty list"
// +openshift:validation:FeatureGateAwareXValidation:featureGate=BootImageSkewEnforcement,rule="self.?status.bootImageSkewEnforcementStatus.mode.orValue(\"\") == 'Automatic' ? !(self.?spec.managedBootImages.machineManagers.hasValue()) || !self.spec.managedBootImages.machineManagers.exists(m, m.resource == 'machinesets' && m.apiGroup == 'machine.openshift.io') || self.spec.managedBootImages.machineManagers.exists(m, m.resource == 'machinesets' && m.apiGroup == 'machine.openshift.io' && m.selection.mode == 'All') : true",message="when skew enforcement is in Automatic mode, any MachineAPI MachineSet MachineManager must use selection mode 'All'"
// +openshift:validation:FeatureGateAwareXValidation:featureGate=BootImageSkewEnforcement,rule="self.?status.bootImageSkewEnforcementStatus.mode.orValue(\"\") == 'Automatic' ? !(self.?status.managedBootImagesStatus.machineManagers.hasValue()) || self.status.managedBootImagesStatus.machineManagers.exists(m, m.selection.mode == 'All' && m.resource == 'machinesets' && m.apiGroup == 'machine.openshift.io'): true",message="when skew enforcement is in Automatic mode, managedBootImagesStatus must contain a MachineManager opting in all MachineAPI MachineSets"
type MachineConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata"`

	// spec is the specification of the desired behavior of the Machine Config Operator
	// +required
	Spec MachineConfigurationSpec `json:"spec"`

	// status is the most recently observed status of the Machine Config Operator
	// +optional
	Status MachineConfigurationStatus `json:"status"`
}

type MachineConfigurationSpec struct {
	StaticPodOperatorSpec `json:",inline"`

	// managedBootImages allows configuration for the management of boot images for machine
	// resources within the cluster. This configuration allows users to select resources that should
	// be updated to the latest boot images during cluster upgrades, ensuring that new machines
	// always boot with the current cluster version's boot image. When omitted, this means no opinion
	// and the platform is left to choose a reasonable default, which is subject to change over time.
	// The default for each machine manager mode is All for GCP and AWS platforms, and None for all
	// other platforms.
	// +optional
	ManagedBootImages ManagedBootImages `json:"managedBootImages"`

	// nodeDisruptionPolicy allows an admin to set granular node disruption actions for
	// MachineConfig-based updates, such as drains, service reloads, etc. Specifying this will allow
	// for less downtime when doing small configuration updates to the cluster. This configuration
	// has no effect on cluster upgrades which will still incur node disruption where required.
	// +optional
	NodeDisruptionPolicy NodeDisruptionPolicyConfig `json:"nodeDisruptionPolicy"`

	// irreconcilableValidationOverrides is an optional field that can used to make changes to a MachineConfig that
	// cannot be applied to existing nodes.
	// When specified, the fields configured with validation overrides will no longer reject changes to those
	// respective fields due to them not being able to be applied to existing nodes.
	// Only newly provisioned nodes will have these configurations applied.
	// Existing nodes will report observed configuration differences in their MachineConfigNode status.
	// +openshift:enable:FeatureGate=IrreconcilableMachineConfig
	// +optional
	IrreconcilableValidationOverrides IrreconcilableValidationOverrides `json:"irreconcilableValidationOverrides,omitempty,omitzero"`

	// bootImageSkewEnforcement allows an admin to configure how boot image version skew is
	// enforced on the cluster.
	// When omitted, this will default to Automatic for clusters that support automatic boot image updates.
	// For clusters that do not support automatic boot image updates, cluster upgrades will be disabled until
	// a skew enforcement mode has been specified.
	// When version skew is being enforced, cluster upgrades will be disabled until the version skew is deemed
	// acceptable for the current release payload.
	// +openshift:enable:FeatureGate=BootImageSkewEnforcement
	// +optional
	BootImageSkewEnforcement BootImageSkewEnforcementConfig `json:"bootImageSkewEnforcement,omitempty,omitzero"`
}

// BootImageSkewEnforcementConfig is used to configure how boot image version skew is enforced on the cluster.
// +kubebuilder:validation:XValidation:rule="has(self.mode) && (self.mode =='Manual') ?  has(self.manual) : !has(self.manual)",message="manual is required when mode is Manual, and forbidden otherwise"
// +union
type BootImageSkewEnforcementConfig struct {
	// mode determines the underlying behavior of skew enforcement mechanism.
	// Valid values are Manual and None.
	// Manual means that the cluster admin is expected to perform manual boot image updates and store the OCP
	// & RHCOS version associated with the last boot image update in the manual field.
	// In Manual mode, the MCO will prevent upgrades when the boot image skew exceeds the
	// skew limit described by the release image.
	// None means that the MCO will no longer monitor the boot image skew. This may affect
	// the cluster's ability to scale.
	// This field is required.
	// +unionDiscriminator
	// +required
	Mode BootImageSkewEnforcementConfigMode `json:"mode,omitempty"`

	// manual describes the current boot image of the cluster.
	// This should be set to the oldest boot image used amongst all machine resources in the cluster.
	// This must include either the RHCOS version of the boot image or the OCP release version which shipped with that
	// RHCOS boot image.
	// Required when mode is set to "Manual" and forbidden otherwise.
	// +optional
	Manual ClusterBootImageManual `json:"manual,omitempty,omitzero"`
}

// ClusterBootImageManual is used to describe the cluster boot image in Manual mode.
// +kubebuilder:validation:XValidation:rule="has(self.mode) && (self.mode =='OCPVersion') ?  has(self.ocpVersion) : !has(self.ocpVersion)",message="ocpVersion is required when mode is OCPVersion, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.mode) && (self.mode =='RHCOSVersion') ?  has(self.rhcosVersion) : !has(self.rhcosVersion)",message="rhcosVersion is required when mode is RHCOSVersion, and forbidden otherwise"
// +union
type ClusterBootImageManual struct {
	// mode is used to configure which boot image field is defined in Manual mode.
	// Valid values are OCPVersion and RHCOSVersion.
	// OCPVersion means that the cluster admin is expected to set the OCP version associated with the last boot image update
	// in the OCPVersion field.
	// RHCOSVersion means that the cluster admin is expected to set the RHCOS version associated with the last boot image update
	// in the RHCOSVersion field.
	// This field is required.
	// +unionDiscriminator
	// +required
	Mode ClusterBootImageManualMode `json:"mode,omitempty"`

	// ocpVersion provides a string which represents the OCP version of the boot image.
	// This field must match the OCP semver compatible format of x.y.z. This field must be between
	// 5 and 10 characters long.
	// Required when mode is set to "OCPVersion" and forbidden otherwise.
	// +kubebuilder:validation:XValidation:rule="self.matches('^[0-9]+\\\\.[0-9]+\\\\.[0-9]+$')",message="ocpVersion must match the OCP semver compatible format of x.y.z"
	// +kubebuilder:validation:MaxLength:=10
	// +kubebuilder:validation:MinLength:=5
	// +optional
	OCPVersion string `json:"ocpVersion,omitempty"`

	// rhcosVersion provides a string which represents the RHCOS version of the boot image
	// This field must match rhcosVersion formatting of [major].[minor].[datestamp(YYYYMMDD)]-[buildnumber] or the legacy
	// format of [major].[minor].[timestamp(YYYYMMDDHHmm)]-[buildnumber]. This field must be between
	// 14 and 21 characters long.
	// Required when mode is set to "RHCOSVersion" and forbidden otherwise.
	// +kubebuilder:validation:XValidation:rule="self.matches('^[0-9]+\\\\.[0-9]+\\\\.([0-9]{8}|[0-9]{12})-[0-9]+$')",message="rhcosVersion must match format [major].[minor].[datestamp(YYYYMMDD)]-[buildnumber] or must match legacy format [major].[minor].[timestamp(YYYYMMDDHHmm)]-[buildnumber]"
	// +kubebuilder:validation:MaxLength:=21
	// +kubebuilder:validation:MinLength:=14
	// +optional
	RHCOSVersion string `json:"rhcosVersion,omitempty"`
}

// ClusterBootImageManualMode is a string enum used to define the cluster's boot image in manual mode.
// +kubebuilder:validation:Enum:="OCPVersion";"RHCOSVersion"
type ClusterBootImageManualMode string

const (
	// OCPVersion represents a configuration mode used to define the OCPVersion.
	ClusterBootImageSpecModeOCPVersion ClusterBootImageManualMode = "OCPVersion"

	// RHCOSVersion represents a configuration mode used to define the RHCOSVersion.
	ClusterBootImageSpecModeRHCOSVersion ClusterBootImageManualMode = "RHCOSVersion"
)

// BootImageSkewEnforcementStatus is the type for the status object. It represents the cluster defaults when
// the boot image skew enforcement configuration is undefined and reflects the actual configuration when it is defined.
// +kubebuilder:validation:XValidation:rule="has(self.mode) && (self.mode == 'Automatic') ?  has(self.automatic) : !has(self.automatic)",message="automatic is required when mode is Automatic, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.mode) && (self.mode == 'Manual') ?  has(self.manual) : !has(self.manual)",message="manual is required when mode is Manual, and forbidden otherwise"
// +union
type BootImageSkewEnforcementStatus struct {
	// mode determines the underlying behavior of skew enforcement mechanism.
	// Valid values are Automatic, Manual and None.
	// Automatic means that the MCO will perform boot image updates and store the
	// OCP & RHCOS version associated with the last boot image update in the automatic field.
	// Manual means that the cluster admin is expected to perform manual boot image updates and store the OCP
	// & RHCOS version associated with the last boot image update in the manual field.
	// In Automatic and Manual mode, the MCO will prevent upgrades when the boot image skew exceeds the
	// skew limit described by the release image.
	// None means that the MCO will no longer monitor the boot image skew. This may affect
	// the cluster's ability to scale.
	// This field is required.
	// +unionDiscriminator
	// +required
	Mode BootImageSkewEnforcementModeStatus `json:"mode,omitempty"`

	// automatic describes the current boot image of the cluster.
	// This will be populated by the MCO when performing boot image updates. This value will be compared against
	// the cluster's skew limit to determine skew compliance.
	// Required when mode is set to "Automatic" and forbidden otherwise.
	// +optional
	Automatic ClusterBootImageAutomatic `json:"automatic,omitempty,omitzero"`

	// manual describes the current boot image of the cluster.
	// This will be populated by the MCO using the values provided in the spec.bootImageSkewEnforcement.manual field.
	// This value will be compared against the cluster's skew limit to determine skew compliance.
	// Required when mode is set to "Manual" and forbidden otherwise.
	// +optional
	Manual ClusterBootImageManual `json:"manual,omitempty,omitzero"`
}

// ClusterBootImageAutomatic is used to describe the cluster boot image in Automatic mode. It stores the RHCOS version of the
// boot image and the OCP release version which shipped with that RHCOS boot image. At least one of these values are required.
// If ocpVersion and rhcosVersion are defined, both values will be used for checking skew compliance.
// If only ocpVersion is defined, only that value will be used for checking skew compliance.
// If only rhcosVersion is defined, only that value will be used for checking skew compliance.
// +kubebuilder:validation:XValidation:rule="has(self.ocpVersion) || has(self.rhcosVersion)",message="at least one of ocpVersion or rhcosVersion is required"
// +kubebuilder:validation:MinProperties=1
type ClusterBootImageAutomatic struct {
	// ocpVersion provides a string which represents the OCP version of the boot image.
	// This field must match the OCP semver compatible format of x.y.z. This field must be between
	// 5 and 10 characters long.
	// +kubebuilder:validation:XValidation:rule="self.matches('^[0-9]+\\\\.[0-9]+\\\\.[0-9]+$')",message="ocpVersion must match the OCP semver compatible format of x.y.z"
	// +kubebuilder:validation:MaxLength:=10
	// +kubebuilder:validation:MinLength:=5
	// +optional
	OCPVersion string `json:"ocpVersion,omitempty"`

	// rhcosVersion provides a string which represents the RHCOS version of the boot image
	// This field must match rhcosVersion formatting of [major].[minor].[datestamp(YYYYMMDD)]-[buildnumber] or the legacy
	// format of [major].[minor].[timestamp(YYYYMMDDHHmm)]-[buildnumber]. This field must be between
	// 14 and 21 characters long.
	// +kubebuilder:validation:XValidation:rule="self.matches('^[0-9]+\\\\.[0-9]+\\\\.([0-9]{8}|[0-9]{12})-[0-9]+$')",message="rhcosVersion must match format [major].[minor].[datestamp(YYYYMMDD)]-[buildnumber] or must match legacy format [major].[minor].[timestamp(YYYYMMDDHHmm)]-[buildnumber]"
	// +kubebuilder:validation:MaxLength:=21
	// +kubebuilder:validation:MinLength:=14
	// +optional
	RHCOSVersion string `json:"rhcosVersion,omitempty"`
}

// BootImageSkewEnforcementConfigMode is a string enum used to configure the cluster's boot image skew enforcement mode.
// +kubebuilder:validation:Enum:="Manual";"None"
type BootImageSkewEnforcementConfigMode string

const (
	// Manual represents a configuration mode that allows manual skew enforcement.
	BootImageSkewEnforcementConfigModeManual BootImageSkewEnforcementConfigMode = "Manual"

	// None represents a configuration mode that disables boot image skew enforcement.
	BootImageSkewEnforcementConfigModeNone BootImageSkewEnforcementConfigMode = "None"
)

// BootImageSkewEnforcementModeStatus is a string enum used to indicate the cluster's boot image skew enforcement mode.
// +kubebuilder:validation:Enum:="Automatic";"Manual";"None"
type BootImageSkewEnforcementModeStatus string

const (
	// Automatic represents a configuration mode that allows automatic skew enforcement.
	BootImageSkewEnforcementModeStatusAutomatic BootImageSkewEnforcementModeStatus = "Automatic"

	// Manual represents a configuration mode that allows manual skew enforcement.
	BootImageSkewEnforcementModeStatusManual BootImageSkewEnforcementModeStatus = "Manual"

	// None represents a configuration mode that disables boot image skew enforcement.
	BootImageSkewEnforcementModeStatusNone BootImageSkewEnforcementModeStatus = "None"
)

type MachineConfigurationStatus struct {
	// observedGeneration is the last generation change you've dealt with
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// conditions is a list of conditions and their status
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// Previously there was a StaticPodOperatorStatus here for legacy reasons. Many of the fields within
	// it are no longer relevant for the MachineConfiguration CRD's functions. The following remainder
	// fields were tombstoned after lifting out StaticPodOperatorStatus. To avoid conflicts with
	// serialisation, the following field names may never be used again.

	// Tombstone: legacy field from StaticPodOperatorStatus
	// Version string `json:"version,omitempty"`

	// Tombstone: legacy field from StaticPodOperatorStatus
	// ReadyReplicas int32 `json:"readyReplicas"`

	// Tombstone: legacy field from StaticPodOperatorStatus
	// Generations []GenerationStatus `json:"generations,omitempty"`

	// Tombstone: legacy field from StaticPodOperatorStatus
	// LatestAvailableRevision int32 `json:"latestAvailableRevision,omitempty"`

	// Tombstone: legacy field from StaticPodOperatorStatus
	// LatestAvailableRevisionReason string `json:"latestAvailableRevisionReason,omitempty"`

	// Tombstone: legacy field from StaticPodOperatorStatus
	// NodeStatuses []NodeStatus `json:"nodeStatuses,omitempty"`

	// nodeDisruptionPolicyStatus status reflects what the latest cluster-validated policies are,
	// and will be used by the Machine Config Daemon during future node updates.
	// +optional
	NodeDisruptionPolicyStatus NodeDisruptionPolicyStatus `json:"nodeDisruptionPolicyStatus"`

	// managedBootImagesStatus reflects what the latest cluster-validated boot image configuration is
	// and will be used by Machine Config Controller while performing boot image updates.
	// +optional
	ManagedBootImagesStatus ManagedBootImages `json:"managedBootImagesStatus"`

	// bootImageSkewEnforcementStatus reflects what the latest cluster-validated boot image skew enforcement
	// configuration is and will be used by Machine Config Controller while performing boot image skew enforcement.
	// When omitted, the MCO has no knowledge of how to enforce boot image skew. When the MCO does not know how
	// boot image skew should be enforced, cluster upgrades will be blocked until it can either automatically
	// determine skew enforcement or there is an explicit skew enforcement configuration provided in the
	// spec.bootImageSkewEnforcement field.
	// +openshift:enable:FeatureGate=BootImageSkewEnforcement
	// +optional
	BootImageSkewEnforcementStatus BootImageSkewEnforcementStatus `json:"bootImageSkewEnforcementStatus,omitempty,omitzero"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// MachineConfigurationList is a collection of items
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type MachineConfigurationList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	// items contains the items
	Items []MachineConfiguration `json:"items"`
}

// IrreconcilableValidationOverridesStorage defines available storage irreconcilable overrides.
// +kubebuilder:validation:Enum=Disks;FileSystems;Raid
type IrreconcilableValidationOverridesStorage string

const (
	// Disks enables changes to the `spec.config.storage.disks` section of MachineConfig CRs.
	IrreconcilableValidationOverridesStorageDisks IrreconcilableValidationOverridesStorage = "Disks"

	// FileSystems enables changes to the `spec.config.storage.filesystems` section of MachineConfig CRs.
	IrreconcilableValidationOverridesStorageFileSystems IrreconcilableValidationOverridesStorage = "FileSystems"

	// Raid enables changes to the `spec.config.storage.raid` section of MachineConfig CRs.
	IrreconcilableValidationOverridesStorageRaid IrreconcilableValidationOverridesStorage = "Raid"
)

// IrreconcilableValidationOverrides holds the irreconcilable validations overrides to be applied on each rendered
// MachineConfig generation.
// +kubebuilder:validation:MinProperties=1
type IrreconcilableValidationOverrides struct {
	// storage can be used to allow making irreconcilable changes to the selected sections under the
	// `spec.config.storage` field of MachineConfig CRs
	// It must have at least one item, may not exceed 3 items and must not contain duplicates.
	// Allowed element values are "Disks", "FileSystems", "Raid" and omitted.
	// When contains "Disks" changes to the `spec.config.storage.disks` section of MachineConfig CRs are allowed.
	// When contains "FileSystems" changes to the `spec.config.storage.filesystems` section of MachineConfig CRs are allowed.
	// When contains "Raid" changes to the `spec.config.storage.raid` section of MachineConfig CRs are allowed.
	// When omitted changes to the `spec.config.storage` section are forbidden.
	// +optional
	// +listType=set
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=3
	Storage []IrreconcilableValidationOverridesStorage `json:"storage,omitempty,omitzero"`
}

type ManagedBootImages struct {
	// machineManagers can be used to register machine management resources for boot image updates. The Machine Config Operator
	// will watch for changes to this list. Only one entry is permitted per type of machine management resource.
	// +optional
	// +listType=map
	// +listMapKey=resource
	// +listMapKey=apiGroup
	// +kubebuilder:validation:MaxItems=5
	MachineManagers []MachineManager `json:"machineManagers"`
}

// MachineManager describes a target machine resource that is registered for boot image updates. It stores identifying information
// such as the resource type and the API Group of the resource. It also provides granular control via the selection field.
// +openshift:validation:FeatureGateAwareXValidation:requiredFeatureGate=ManagedBootImagesCPMS,rule="self.resource != 'controlplanemachinesets' || self.selection.mode == 'All' || self.selection.mode == 'None'", message="Only All or None selection mode is permitted for ControlPlaneMachineSets"
type MachineManager struct {
	// resource is the machine management resource's type.
	// Valid values are machinesets and controlplanemachinesets.
	// machinesets means that the machine manager will only register resources of the kind MachineSet.
	// controlplanemachinesets means that the machine manager will only register resources of the kind ControlPlaneMachineSet.
	// +required
	Resource MachineManagerMachineSetsResourceType `json:"resource"`

	// apiGroup is name of the APIGroup that the machine management resource belongs to.
	// The only current valid value is machine.openshift.io.
	// machine.openshift.io means that the machine manager will only register resources that belong to OpenShift machine API group.
	// +required
	APIGroup MachineManagerMachineSetsAPIGroupType `json:"apiGroup"`

	// selection allows granular control of the machine management resources that will be registered for boot image updates.
	// +required
	Selection MachineManagerSelector `json:"selection"`
}

// +kubebuilder:validation:XValidation:rule="has(self.mode) && self.mode == 'Partial' ?  has(self.partial) : !has(self.partial)",message="Partial is required when type is partial, and forbidden otherwise"
// +union
type MachineManagerSelector struct {
	// mode determines how machine managers will be selected for updates.
	// Valid values are All, Partial and None.
	// All means that every resource matched by the machine manager will be updated.
	// Partial requires specified selector(s) and allows customisation of which resources matched by the machine manager will be updated.
	// Partial is not permitted for the controlplanemachinesets resource type as they are a singleton within the cluster.
	// None means that every resource matched by the machine manager will not be updated.
	// +unionDiscriminator
	// +required
	Mode MachineManagerSelectorMode `json:"mode"`

	// partial provides label selector(s) that can be used to match machine management resources.
	// Only permitted when mode is set to "Partial".
	// +optional
	Partial *PartialSelector `json:"partial,omitempty"`
}

// PartialSelector provides label selector(s) that can be used to match machine management resources.
type PartialSelector struct {
	// machineResourceSelector is a label selector that can be used to select machine resources like MachineSets.
	// +required
	MachineResourceSelector *metav1.LabelSelector `json:"machineResourceSelector,omitempty"`
}

// MachineManagerSelectorMode is a string enum used in the MachineManagerSelector union discriminator.
// +kubebuilder:validation:Enum:="All";"Partial";"None"
type MachineManagerSelectorMode string

const (
	// All represents a configuration mode that registers all resources specified by the parent MachineManager for boot image updates.
	All MachineManagerSelectorMode = "All"

	// Partial represents a configuration mode that will register resources specified by the parent MachineManager only
	// if they match with the label selector.
	Partial MachineManagerSelectorMode = "Partial"

	// None represents a configuration mode that excludes all resources specified by the parent MachineManager from boot image updates.
	None MachineManagerSelectorMode = "None"
)

// MachineManagerManagedResourceType is a string enum used in the MachineManager type to describe the resource
// type to be registered.
// +openshift:validation:FeatureGateAwareEnum:featureGate="",enum=machinesets
// +openshift:validation:FeatureGateAwareEnum:featureGate=ManagedBootImagesCPMS,enum=machinesets;controlplanemachinesets
type MachineManagerMachineSetsResourceType string

const (
	// MachineSets represent the MachineSet resource type, which manage a group of machines and belong to the Openshift machine API group.
	MachineSets MachineManagerMachineSetsResourceType = "machinesets"
	// ControlPlaneMachineSets represent the ControlPlaneMachineSets resource type, which manage a group of control-plane machines and belong to the Openshift machine API group.
	ControlPlaneMachineSets MachineManagerMachineSetsResourceType = "controlplanemachinesets"
)

// MachineManagerManagedAPIGroupType is a string enum used in in the MachineManager type to describe the APIGroup
// of the resource type being registered.
// +kubebuilder:validation:Enum:="machine.openshift.io"
type MachineManagerMachineSetsAPIGroupType string

const (
	// MachineAPI represent the traditional MAPI Group that a machineset may belong to.
	// This feature only supports MAPI machinesets and controlplanemachinesets at this time.
	MachineAPI MachineManagerMachineSetsAPIGroupType = "machine.openshift.io"
)

type NodeDisruptionPolicyStatus struct {
	// clusterPolicies is a merge of cluster default and user provided node disruption policies.
	// +optional
	ClusterPolicies NodeDisruptionPolicyClusterStatus `json:"clusterPolicies"`
}

// NodeDisruptionPolicyConfig is the overall spec definition for files/units/sshkeys
type NodeDisruptionPolicyConfig struct {
	// files is a list of MachineConfig file definitions and actions to take to changes on those paths
	// This list supports a maximum of 50 entries.
	// +optional
	// +listType=map
	// +listMapKey=path
	// +kubebuilder:validation:MaxItems=50
	Files []NodeDisruptionPolicySpecFile `json:"files"`
	// units is a list MachineConfig unit definitions and actions to take on changes to those services
	// This list supports a maximum of 50 entries.
	// +optional
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MaxItems=50
	Units []NodeDisruptionPolicySpecUnit `json:"units"`
	// sshkey maps to the ignition.sshkeys field in the MachineConfig object, definition an action for this
	// will apply to all sshkey changes in the cluster
	// +optional
	SSHKey NodeDisruptionPolicySpecSSHKey `json:"sshkey"`
}

// NodeDisruptionPolicyClusterStatus is the type for the status object, rendered by the controller as a
// merge of cluster defaults and user provided policies
type NodeDisruptionPolicyClusterStatus struct {
	// files is a list of MachineConfig file definitions and actions to take to changes on those paths
	// +optional
	// +listType=map
	// +listMapKey=path
	// +kubebuilder:validation:MaxItems=100
	Files []NodeDisruptionPolicyStatusFile `json:"files,omitempty"`
	// units is a list MachineConfig unit definitions and actions to take on changes to those services
	// +optional
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MaxItems=100
	Units []NodeDisruptionPolicyStatusUnit `json:"units,omitempty"`
	// sshkey is the overall sshkey MachineConfig definition
	// +optional
	SSHKey NodeDisruptionPolicyStatusSSHKey `json:"sshkey,omitempty"`
}

// NodeDisruptionPolicySpecFile is a file entry and corresponding actions to take and is used in the NodeDisruptionPolicyConfig object
type NodeDisruptionPolicySpecFile struct {
	// path is the location of a file being managed through a MachineConfig.
	// The Actions in the policy will apply to changes to the file at this path.
	// +required
	Path string `json:"path"`
	// actions represents the series of commands to be executed on changes to the file at
	// the corresponding file path. Actions will be applied in the order that
	// they are set in this list. If there are other incoming changes to other MachineConfig
	// entries in the same update that require a reboot, the reboot will supercede these actions.
	// Valid actions are Reboot, Drain, Reload, DaemonReload and None.
	// The Reboot action and the None action cannot be used in conjunction with any of the other actions.
	// This list supports a maximum of 10 entries.
	// +required
	// +listType=atomic
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='Reboot') ? size(self) == 1 : true", message="Reboot action can only be specified standalone, as it will override any other actions"
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='None') ? size(self) == 1 : true", message="None action can only be specified standalone, as it will override any other actions"
	Actions []NodeDisruptionPolicySpecAction `json:"actions"`
}

// NodeDisruptionPolicyStatusFile is a file entry and corresponding actions to take and is used in the NodeDisruptionPolicyClusterStatus object
type NodeDisruptionPolicyStatusFile struct {
	// path is the location of a file being managed through a MachineConfig.
	// The Actions in the policy will apply to changes to the file at this path.
	// +required
	Path string `json:"path"`
	// actions represents the series of commands to be executed on changes to the file at
	// the corresponding file path. Actions will be applied in the order that
	// they are set in this list. If there are other incoming changes to other MachineConfig
	// entries in the same update that require a reboot, the reboot will supercede these actions.
	// Valid actions are Reboot, Drain, Reload, DaemonReload and None.
	// The Reboot action and the None action cannot be used in conjunction with any of the other actions.
	// This list supports a maximum of 10 entries.
	// +required
	// +listType=atomic
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='Reboot') ? size(self) == 1 : true", message="Reboot action can only be specified standalone, as it will override any other actions"
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='None') ? size(self) == 1 : true", message="None action can only be specified standalone, as it will override any other actions"
	Actions []NodeDisruptionPolicyStatusAction `json:"actions"`
}

// NodeDisruptionPolicySpecUnit is a systemd unit name and corresponding actions to take and is used in the NodeDisruptionPolicyConfig object
type NodeDisruptionPolicySpecUnit struct {
	// name represents the service name of a systemd service managed through a MachineConfig
	// Actions specified will be applied for changes to the named service.
	// Service names should be of the format ${NAME}${SERVICETYPE} and can up to 255 characters long.
	// ${NAME} must be atleast 1 character long and can only consist of alphabets, digits, ":", "-", "_", ".", and "\".
	// ${SERVICETYPE} must be one of ".service", ".socket", ".device", ".mount", ".automount", ".swap", ".target", ".path", ".timer", ".snapshot", ".slice" or ".scope".
	// +required
	Name NodeDisruptionPolicyServiceName `json:"name"`

	// actions represents the series of commands to be executed on changes to the file at
	// the corresponding file path. Actions will be applied in the order that
	// they are set in this list. If there are other incoming changes to other MachineConfig
	// entries in the same update that require a reboot, the reboot will supercede these actions.
	// Valid actions are Reboot, Drain, Reload, DaemonReload and None.
	// The Reboot action and the None action cannot be used in conjunction with any of the other actions.
	// This list supports a maximum of 10 entries.
	// +required
	// +listType=atomic
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='Reboot') ? size(self) == 1 : true", message="Reboot action can only be specified standalone, as it will override any other actions"
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='None') ? size(self) == 1 : true", message="None action can only be specified standalone, as it will override any other actions"
	Actions []NodeDisruptionPolicySpecAction `json:"actions"`
}

// NodeDisruptionPolicyStatusUnit is a systemd unit name and corresponding actions to take and is used in the NodeDisruptionPolicyClusterStatus object
type NodeDisruptionPolicyStatusUnit struct {
	// name represents the service name of a systemd service managed through a MachineConfig
	// Actions specified will be applied for changes to the named service.
	// Service names should be of the format ${NAME}${SERVICETYPE} and can up to 255 characters long.
	// ${NAME} must be atleast 1 character long and can only consist of alphabets, digits, ":", "-", "_", ".", and "\".
	// ${SERVICETYPE} must be one of ".service", ".socket", ".device", ".mount", ".automount", ".swap", ".target", ".path", ".timer", ".snapshot", ".slice" or ".scope".
	// +required
	Name NodeDisruptionPolicyServiceName `json:"name"`

	// actions represents the series of commands to be executed on changes to the file at
	// the corresponding file path. Actions will be applied in the order that
	// they are set in this list. If there are other incoming changes to other MachineConfig
	// entries in the same update that require a reboot, the reboot will supercede these actions.
	// Valid actions are Reboot, Drain, Reload, DaemonReload and None.
	// The Reboot action and the None action cannot be used in conjunction with any of the other actions.
	// This list supports a maximum of 10 entries.
	// +required
	// +listType=atomic
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='Reboot') ? size(self) == 1 : true", message="Reboot action can only be specified standalone, as it will override any other actions"
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='None') ? size(self) == 1 : true", message="None action can only be specified standalone, as it will override any other actions"
	Actions []NodeDisruptionPolicyStatusAction `json:"actions"`
}

// NodeDisruptionPolicySpecSSHKey is actions to take for any SSHKey change and is used in the NodeDisruptionPolicyConfig object
type NodeDisruptionPolicySpecSSHKey struct {
	// actions represents the series of commands to be executed on changes to the file at
	// the corresponding file path. Actions will be applied in the order that
	// they are set in this list. If there are other incoming changes to other MachineConfig
	// entries in the same update that require a reboot, the reboot will supercede these actions.
	// Valid actions are Reboot, Drain, Reload, DaemonReload and None.
	// The Reboot action and the None action cannot be used in conjunction with any of the other actions.
	// This list supports a maximum of 10 entries.
	// +required
	// +listType=atomic
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='Reboot') ? size(self) == 1 : true", message="Reboot action can only be specified standalone, as it will override any other actions"
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='None') ? size(self) == 1 : true", message="None action can only be specified standalone, as it will override any other actions"
	Actions []NodeDisruptionPolicySpecAction `json:"actions"`
}

// NodeDisruptionPolicyStatusSSHKey is actions to take for any SSHKey change and is used in the NodeDisruptionPolicyClusterStatus object
type NodeDisruptionPolicyStatusSSHKey struct {
	// actions represents the series of commands to be executed on changes to the file at
	// the corresponding file path. Actions will be applied in the order that
	// they are set in this list. If there are other incoming changes to other MachineConfig
	// entries in the same update that require a reboot, the reboot will supercede these actions.
	// Valid actions are Reboot, Drain, Reload, DaemonReload and None.
	// The Reboot action and the None action cannot be used in conjunction with any of the other actions.
	// This list supports a maximum of 10 entries.
	// +required
	// +listType=atomic
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='Reboot') ? size(self) == 1 : true", message="Reboot action can only be specified standalone, as it will override any other actions"
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='None') ? size(self) == 1 : true", message="None action can only be specified standalone, as it will override any other actions"
	Actions []NodeDisruptionPolicyStatusAction `json:"actions"`
}

// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'Reload' ? has(self.reload) : !has(self.reload)",message="reload is required when type is Reload, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'Restart' ? has(self.restart) : !has(self.restart)",message="restart is required when type is Restart, and forbidden otherwise"
// +union
type NodeDisruptionPolicySpecAction struct {
	// type represents the commands that will be carried out if this NodeDisruptionPolicySpecActionType is executed
	// Valid values are Reboot, Drain, Reload, Restart, DaemonReload and None.
	// reload/restart requires a corresponding service target specified in the reload/restart field.
	// Other values require no further configuration
	// +unionDiscriminator
	// +required
	Type NodeDisruptionPolicySpecActionType `json:"type"`
	// reload specifies the service to reload, only valid if type is reload
	// +optional
	Reload *ReloadService `json:"reload,omitempty"`
	// restart specifies the service to restart, only valid if type is restart
	// +optional
	Restart *RestartService `json:"restart,omitempty"`
}

// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'Reload' ? has(self.reload) : !has(self.reload)",message="reload is required when type is Reload, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'Restart' ? has(self.restart) : !has(self.restart)",message="restart is required when type is Restart, and forbidden otherwise"
// +union
type NodeDisruptionPolicyStatusAction struct {
	// type represents the commands that will be carried out if this NodeDisruptionPolicyStatusActionType is executed
	// Valid values are Reboot, Drain, Reload, Restart, DaemonReload, None and Special.
	// reload/restart requires a corresponding service target specified in the reload/restart field.
	// Other values require no further configuration
	// +unionDiscriminator
	// +required
	Type NodeDisruptionPolicyStatusActionType `json:"type"`
	// reload specifies the service to reload, only valid if type is reload
	// +optional
	Reload *ReloadService `json:"reload,omitempty"`
	// restart specifies the service to restart, only valid if type is restart
	// +optional
	Restart *RestartService `json:"restart,omitempty"`
}

// ReloadService allows the user to specify the services to be reloaded
type ReloadService struct {
	// serviceName is the full name (e.g. crio.service) of the service to be reloaded
	// Service names should be of the format ${NAME}${SERVICETYPE} and can up to 255 characters long.
	// ${NAME} must be atleast 1 character long and can only consist of alphabets, digits, ":", "-", "_", ".", and "\".
	// ${SERVICETYPE} must be one of ".service", ".socket", ".device", ".mount", ".automount", ".swap", ".target", ".path", ".timer", ".snapshot", ".slice" or ".scope".
	// +required
	ServiceName NodeDisruptionPolicyServiceName `json:"serviceName"`
}

// RestartService allows the user to specify the services to be restarted
type RestartService struct {
	// serviceName is the full name (e.g. crio.service) of the service to be restarted
	// Service names should be of the format ${NAME}${SERVICETYPE} and can up to 255 characters long.
	// ${NAME} must be atleast 1 character long and can only consist of alphabets, digits, ":", "-", "_", ".", and "\".
	// ${SERVICETYPE} must be one of ".service", ".socket", ".device", ".mount", ".automount", ".swap", ".target", ".path", ".timer", ".snapshot", ".slice" or ".scope".
	// +required
	ServiceName NodeDisruptionPolicyServiceName `json:"serviceName"`
}

// NodeDisruptionPolicySpecActionType is a string enum used in a NodeDisruptionPolicySpecAction object. They describe an action to be performed.
// +kubebuilder:validation:Enum:="Reboot";"Drain";"Reload";"Restart";"DaemonReload";"None"
type NodeDisruptionPolicySpecActionType string

// +kubebuilder:validation:XValidation:rule=`self.matches('\\.(service|socket|device|mount|automount|swap|target|path|timer|snapshot|slice|scope)$')`, message="Invalid ${SERVICETYPE} in service name. Expected format is ${NAME}${SERVICETYPE}, where ${SERVICETYPE} must be one of \".service\", \".socket\", \".device\", \".mount\", \".automount\", \".swap\", \".target\", \".path\", \".timer\",\".snapshot\", \".slice\" or \".scope\"."
// +kubebuilder:validation:XValidation:rule=`self.matches('^[a-zA-Z0-9:._\\\\-]+\\..')`, message="Invalid ${NAME} in service name. Expected format is ${NAME}${SERVICETYPE}, where {NAME} must be atleast 1 character long and can only consist of alphabets, digits, \":\", \"-\", \"_\", \".\", and \"\\\""
// +kubebuilder:validation:MaxLength=255
type NodeDisruptionPolicyServiceName string

const (
	// Reboot represents an action that will cause nodes to be rebooted. This is the default action by the MCO
	// if a reboot policy is not found for a change/update being performed by the MCO.
	RebootSpecAction NodeDisruptionPolicySpecActionType = "Reboot"

	// Drain represents an action that will cause nodes to be drained of their workloads.
	DrainSpecAction NodeDisruptionPolicySpecActionType = "Drain"

	// Reload represents an action that will cause nodes to reload the service described by the Target field.
	ReloadSpecAction NodeDisruptionPolicySpecActionType = "Reload"

	// Restart represents an action that will cause nodes to restart the service described by the Target field.
	RestartSpecAction NodeDisruptionPolicySpecActionType = "Restart"

	// DaemonReload represents an action that TBD
	DaemonReloadSpecAction NodeDisruptionPolicySpecActionType = "DaemonReload"

	// None represents an action that no handling is required by the MCO.
	NoneSpecAction NodeDisruptionPolicySpecActionType = "None"
)

// NodeDisruptionPolicyStatusActionType is a string enum used in a NodeDisruptionPolicyStatusAction object. They describe an action to be performed.
// The key difference of this object from NodeDisruptionPolicySpecActionType is that there is a additional SpecialStatusAction value in this enum. This will only be
// used by the MCO's controller to indicate some internal actions. They are not part of the NodeDisruptionPolicyConfig object and cannot be set by the user.
// +kubebuilder:validation:Enum:="Reboot";"Drain";"Reload";"Restart";"DaemonReload";"None";"Special"
type NodeDisruptionPolicyStatusActionType string

const (
	// Reboot represents an action that will cause nodes to be rebooted. This is the default action by the MCO
	// if a reboot policy is not found for a change/update being performed by the MCO.
	RebootStatusAction NodeDisruptionPolicyStatusActionType = "Reboot"

	// Drain represents an action that will cause nodes to be drained of their workloads.
	DrainStatusAction NodeDisruptionPolicyStatusActionType = "Drain"

	// Reload represents an action that will cause nodes to reload the service described by the Target field.
	ReloadStatusAction NodeDisruptionPolicyStatusActionType = "Reload"

	// Restart represents an action that will cause nodes to restart the service described by the Target field.
	RestartStatusAction NodeDisruptionPolicyStatusActionType = "Restart"

	// DaemonReload represents an action that TBD
	DaemonReloadStatusAction NodeDisruptionPolicyStatusActionType = "DaemonReload"

	// None represents an action that no handling is required by the MCO.
	NoneStatusAction NodeDisruptionPolicyStatusActionType = "None"

	// Special represents an action that is internal to the MCO, and is not allowed in user defined NodeDisruption policies.
	SpecialStatusAction NodeDisruptionPolicyStatusActionType = "Special"
)

// These strings will be used for MachineConfiguration Status conditions.
const (
	// MachineConfigurationBootImageUpdateDegraded means that the MCO ran into an error while reconciling boot images. This
	// will cause the clusteroperators.config.openshift.io/machine-config to degrade. This  condition will indicate the cause
	// of the degrade, the progress of the update and the generation of the boot images configmap that it degraded on.
	MachineConfigurationBootImageUpdateDegraded string = "BootImageUpdateDegraded"

	// MachineConfigurationBootImageUpdateProgressing means that the MCO is in the process of reconciling boot images. This
	// will cause the clusteroperators.config.openshift.io/machine-config to be in a Progressing state. This condition will
	// indicate the progress of the update and the generation of the boot images configmap that triggered this update.
	MachineConfigurationBootImageUpdateProgressing string = "BootImageUpdateProgressing"
)
