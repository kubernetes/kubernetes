package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ClusterVersion is the configuration for the ClusterVersionOperator. This is where
// parameters related to automatic updates can be set.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/495
// +openshift:file-pattern=cvoRunLevel=0000_00,operatorName=cluster-version-operator,operatorOrdering=01
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:path=clusterversions,scope=Cluster
// +kubebuilder:validation:XValidation:rule="has(self.spec.capabilities) && has(self.spec.capabilities.additionalEnabledCapabilities) && self.spec.capabilities.baselineCapabilitySet == 'None' && 'marketplace' in self.spec.capabilities.additionalEnabledCapabilities ? 'OperatorLifecycleManager' in self.spec.capabilities.additionalEnabledCapabilities || (has(self.status) && has(self.status.capabilities) && has(self.status.capabilities.enabledCapabilities) && 'OperatorLifecycleManager' in self.status.capabilities.enabledCapabilities) : true",message="the `marketplace` capability requires the `OperatorLifecycleManager` capability, which is neither explicitly or implicitly enabled in this cluster, please enable the `OperatorLifecycleManager` capability"
// +kubebuilder:printcolumn:name=Version,JSONPath=.status.history[?(@.state=="Completed")].version,type=string
// +kubebuilder:printcolumn:name=Available,JSONPath=.status.conditions[?(@.type=="Available")].status,type=string
// +kubebuilder:printcolumn:name=Progressing,JSONPath=.status.conditions[?(@.type=="Progressing")].status,type=string
// +kubebuilder:printcolumn:name=Since,JSONPath=.status.conditions[?(@.type=="Progressing")].lastTransitionTime,type=date
// +kubebuilder:printcolumn:name=Status,JSONPath=.status.conditions[?(@.type=="Progressing")].message,type=string
// +kubebuilder:metadata:annotations=include.release.openshift.io/self-managed-high-availability=true
type ClusterVersion struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec is the desired state of the cluster version - the operator will work
	// to ensure that the desired version is applied to the cluster.
	// +required
	Spec ClusterVersionSpec `json:"spec"`
	// status contains information about the available updates and any in-progress
	// updates.
	// +optional
	Status ClusterVersionStatus `json:"status"`
}

// ClusterVersionSpec is the desired version state of the cluster. It includes
// the version the cluster should be at, how the cluster is identified, and
// where the cluster should look for version updates.
// +k8s:deepcopy-gen=true
type ClusterVersionSpec struct {
	// clusterID uniquely identifies this cluster. This is expected to be
	// an RFC4122 UUID value (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx in
	// hexadecimal values). This is a required field.
	// +required
	ClusterID ClusterID `json:"clusterID"`

	// desiredUpdate is an optional field that indicates the desired value of
	// the cluster version. Setting this value will trigger an upgrade (if
	// the current version does not match the desired version). The set of
	// recommended update values is listed as part of available updates in
	// status, and setting values outside that range may cause the upgrade
	// to fail.
	//
	// Some of the fields are inter-related with restrictions and meanings described here.
	// 1. image is specified, version is specified, architecture is specified. API validation error.
	// 2. image is specified, version is specified, architecture is not specified. The version extracted from the referenced image must match the specified version.
	// 3. image is specified, version is not specified, architecture is specified. API validation error.
	// 4. image is specified, version is not specified, architecture is not specified. image is used.
	// 5. image is not specified, version is specified, architecture is specified. version and desired architecture are used to select an image.
	// 6. image is not specified, version is specified, architecture is not specified. version and current architecture are used to select an image.
	// 7. image is not specified, version is not specified, architecture is specified. API validation error.
	// 8. image is not specified, version is not specified, architecture is not specified. API validation error.
	//
	// If an upgrade fails the operator will halt and report status
	// about the failing component. Setting the desired update value back to
	// the previous version will cause a rollback to be attempted. Not all
	// rollbacks will succeed.
	//
	// +optional
	DesiredUpdate *Update `json:"desiredUpdate,omitempty"`

	// upstream may be used to specify the preferred update server. By default
	// it will use the appropriate update server for the cluster and region.
	//
	// +optional
	Upstream URL `json:"upstream,omitempty"`
	// channel is an identifier for explicitly requesting a non-default set
	// of updates to be applied to this cluster. The default channel will
	// contain stable updates that are appropriate for production clusters.
	//
	// +optional
	Channel string `json:"channel,omitempty"`

	// capabilities configures the installation of optional, core
	// cluster components.  A null value here is identical to an
	// empty object; see the child properties for default semantics.
	// +optional
	Capabilities *ClusterVersionCapabilitiesSpec `json:"capabilities,omitempty"`

	// signatureStores contains the upstream URIs to verify release signatures and optional
	// reference to a config map by name containing the PEM-encoded CA bundle.
	//
	// By default, CVO will use existing signature stores if this property is empty.
	// The CVO will check the release signatures in the local ConfigMaps first. It will search for a valid signature
	// in these stores in parallel only when local ConfigMaps did not include a valid signature.
	// Validation will fail if none of the signature stores reply with valid signature before timeout.
	// Setting signatureStores will replace the default signature stores with custom signature stores.
	// Default stores can be used with custom signature stores by adding them manually.
	//
	// A maximum of 32 signature stores may be configured.
	// +kubebuilder:validation:MaxItems=32
	// +openshift:enable:FeatureGate=SignatureStores
	// +listType=map
	// +listMapKey=url
	// +optional
	SignatureStores []SignatureStore `json:"signatureStores"`

	// overrides is list of overides for components that are managed by
	// cluster version operator. Marking a component unmanaged will prevent
	// the operator from creating or updating the object.
	// +listType=map
	// +listMapKey=kind
	// +listMapKey=group
	// +listMapKey=namespace
	// +listMapKey=name
	// +optional
	Overrides []ComponentOverride `json:"overrides,omitempty"`
}

// ClusterVersionStatus reports the status of the cluster versioning,
// including any upgrades that are in progress. The current field will
// be set to whichever version the cluster is reconciling to, and the
// conditions array will report whether the update succeeded, is in
// progress, or is failing.
// +k8s:deepcopy-gen=true
type ClusterVersionStatus struct {
	// desired is the version that the cluster is reconciling towards.
	// If the cluster is not yet fully initialized desired will be set
	// with the information available, which may be an image or a tag.
	// +required
	Desired Release `json:"desired"`

	// history contains a list of the most recent versions applied to the cluster.
	// This value may be empty during cluster startup, and then will be updated
	// when a new update is being applied. The newest update is first in the
	// list and it is ordered by recency. Updates in the history have state
	// Completed if the rollout completed - if an update was failing or halfway
	// applied the state will be Partial. Only a limited amount of update history
	// is preserved.
	// +listType=atomic
	// +optional
	History []UpdateHistory `json:"history,omitempty"`

	// observedGeneration reports which version of the spec is being synced.
	// If this value is not equal to metadata.generation, then the desired
	// and conditions fields may represent a previous version.
	// +required
	ObservedGeneration int64 `json:"observedGeneration"`

	// versionHash is a fingerprint of the content that the cluster will be
	// updated with. It is used by the operator to avoid unnecessary work
	// and is for internal use only.
	// +required
	VersionHash string `json:"versionHash"`

	// capabilities describes the state of optional, core cluster components.
	// +optional
	Capabilities ClusterVersionCapabilitiesStatus `json:"capabilities"`

	// conditions provides information about the cluster version. The condition
	// "Available" is set to true if the desiredUpdate has been reached. The
	// condition "Progressing" is set to true if an update is being applied.
	// The condition "Degraded" is set to true if an update is currently blocked
	// by a temporary or permanent error. Conditions are only valid for the
	// current desiredUpdate when metadata.generation is equal to
	// status.generation.
	// +listType=map
	// +listMapKey=type
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +optional
	Conditions []ClusterOperatorStatusCondition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`

	// availableUpdates contains updates recommended for this
	// cluster. Updates which appear in conditionalUpdates but not in
	// availableUpdates may expose this cluster to known issues. This list
	// may be empty if no updates are recommended, if the update service
	// is unavailable, or if an invalid channel has been specified.
	// +nullable
	// +listType=atomic
	// +required
	AvailableUpdates []Release `json:"availableUpdates"`

	// conditionalUpdates contains the list of updates that may be
	// recommended for this cluster if it meets specific required
	// conditions. Consumers interested in the set of updates that are
	// actually recommended for this cluster should use
	// availableUpdates. This list may be empty if no updates are
	// recommended, if the update service is unavailable, or if an empty
	// or invalid channel has been specified.
	// +listType=atomic
	// +optional
	ConditionalUpdates []ConditionalUpdate `json:"conditionalUpdates,omitempty"`
}

// UpdateState is a constant representing whether an update was successfully
// applied to the cluster or not.
type UpdateState string

const (
	// CompletedUpdate indicates an update was successfully applied
	// to the cluster (all resource updates were successful).
	CompletedUpdate UpdateState = "Completed"
	// PartialUpdate indicates an update was never completely applied
	// or is currently being applied.
	PartialUpdate UpdateState = "Partial"
)

// UpdateHistory is a single attempted update to the cluster.
type UpdateHistory struct {
	// state reflects whether the update was fully applied. The Partial state
	// indicates the update is not fully applied, while the Completed state
	// indicates the update was successfully rolled out at least once (all
	// parts of the update successfully applied).
	// +required
	State UpdateState `json:"state"`

	// startedTime is the time at which the update was started.
	// +required
	StartedTime metav1.Time `json:"startedTime"`

	// completionTime, if set, is when the update was fully applied. The update
	// that is currently being applied will have a null completion time.
	// Completion time will always be set for entries that are not the current
	// update (usually to the started time of the next update).
	// +required
	// +nullable
	CompletionTime *metav1.Time `json:"completionTime"`

	// version is a semantic version identifying the update version. If the
	// requested image does not define a version, or if a failure occurs
	// retrieving the image, this value may be empty.
	//
	// +optional
	Version string `json:"version"`

	// image is a container image location that contains the update. This value
	// is always populated.
	// +required
	Image string `json:"image"`

	// verified indicates whether the provided update was properly verified
	// before it was installed. If this is false the cluster may not be trusted.
	// Verified does not cover upgradeable checks that depend on the cluster
	// state at the time when the update target was accepted.
	// +required
	Verified bool `json:"verified"`

	// acceptedRisks records risks which were accepted to initiate the update.
	// For example, it may menition an Upgradeable=False or missing signature
	// that was overriden via desiredUpdate.force, or an update that was
	// initiated despite not being in the availableUpdates set of recommended
	// update targets.
	// +optional
	AcceptedRisks string `json:"acceptedRisks,omitempty"`
}

// ClusterID is string RFC4122 uuid.
type ClusterID string

// ClusterVersionArchitecture enumerates valid cluster architectures.
// +kubebuilder:validation:Enum="Multi";""
type ClusterVersionArchitecture string

const (
	// ClusterVersionArchitectureMulti identifies a multi architecture. A multi
	// architecture cluster is capable of running nodes with multiple architectures.
	ClusterVersionArchitectureMulti ClusterVersionArchitecture = "Multi"
)

// ClusterVersionCapability enumerates optional, core cluster components.
// +kubebuilder:validation:Enum=openshift-samples;baremetal;marketplace;Console;Insights;Storage;CSISnapshot;NodeTuning;MachineAPI;Build;DeploymentConfig;ImageRegistry;OperatorLifecycleManager;CloudCredential;Ingress;CloudControllerManager;OperatorLifecycleManagerV1
type ClusterVersionCapability string

const (
	// ClusterVersionCapabilityOpenShiftSamples manages the sample
	// image streams and templates stored in the openshift
	// namespace, and any registry credentials, stored as a secret,
	// needed for the image streams to import the images they
	// reference.
	ClusterVersionCapabilityOpenShiftSamples ClusterVersionCapability = "openshift-samples"

	// ClusterVersionCapabilityBaremetal manages the cluster
	// baremetal operator which is responsible for running the metal3
	// deployment.
	ClusterVersionCapabilityBaremetal ClusterVersionCapability = "baremetal"

	// ClusterVersionCapabilityMarketplace manages the Marketplace operator which
	// supplies Operator Lifecycle Manager (OLM) users with default catalogs of
	// "optional" operators.
	//
	// Note that Marketplace has a hard requirement on OLM. OLM can not be disabled
	// while Marketplace is enabled.
	ClusterVersionCapabilityMarketplace ClusterVersionCapability = "marketplace"

	// ClusterVersionCapabilityConsole manages the Console operator which
	// installs and maintains the web console.
	ClusterVersionCapabilityConsole ClusterVersionCapability = "Console"

	// ClusterVersionCapabilityInsights manages the Insights operator which
	// collects anonymized information about the cluster to generate
	// recommendations for possible cluster issues.
	ClusterVersionCapabilityInsights ClusterVersionCapability = "Insights"

	// ClusterVersionCapabilityStorage manages the storage operator which
	// is responsible for providing cluster-wide storage defaults
	// WARNING: Do not disable this capability when deployed to
	// RHEV and OpenStack without reading the docs.
	// These clusters heavily rely on that capability and may cause
	// damage to the cluster.
	ClusterVersionCapabilityStorage ClusterVersionCapability = "Storage"

	// ClusterVersionCapabilityCSISnapshot manages the csi snapshot
	// controller operator which is responsible for watching the
	// VolumeSnapshot CRD objects and manages the creation and deletion
	// lifecycle of volume snapshots
	ClusterVersionCapabilityCSISnapshot ClusterVersionCapability = "CSISnapshot"

	// ClusterVersionCapabilityNodeTuning manages the Node Tuning Operator
	// which is responsible for watching the Tuned and Profile CRD
	// objects and manages the containerized TuneD daemon which controls
	// system level tuning of Nodes
	ClusterVersionCapabilityNodeTuning ClusterVersionCapability = "NodeTuning"

	// ClusterVersionCapabilityMachineAPI manages
	// machine-api-operator
	// cluster-autoscaler-operator
	// cluster-control-plane-machine-set-operator
	// which is responsible for machines configuration and heavily
	// targeted for SNO clusters.
	//
	// The following CRDs are disabled as well
	// machines
	// machineset
	// controlplanemachineset
	//
	// WARNING: Do not disable that capability without reading
	// documentation. This is important part of openshift system
	// and may cause cluster damage
	ClusterVersionCapabilityMachineAPI ClusterVersionCapability = "MachineAPI"

	// ClusterVersionCapabilityBuild manages the Build API which is responsible
	// for watching the Build API objects and managing their lifecycle.
	// The functionality is located under openshift-apiserver and openshift-controller-manager.
	//
	// The following resources are taken into account:
	// - builds
	// - buildconfigs
	ClusterVersionCapabilityBuild ClusterVersionCapability = "Build"

	// ClusterVersionCapabilityDeploymentConfig manages the DeploymentConfig API
	// which is responsible for watching the DeploymentConfig API and managing their lifecycle.
	// The functionality is located under openshift-apiserver and openshift-controller-manager.
	//
	// The following resources are taken into account:
	// - deploymentconfigs
	ClusterVersionCapabilityDeploymentConfig ClusterVersionCapability = "DeploymentConfig"

	// ClusterVersionCapabilityImageRegistry manages the image registry which
	// allows to distribute Docker images
	ClusterVersionCapabilityImageRegistry ClusterVersionCapability = "ImageRegistry"

	// ClusterVersionCapabilityOperatorLifecycleManager manages the Operator Lifecycle Manager (legacy)
	// which itself manages the lifecycle of operators
	ClusterVersionCapabilityOperatorLifecycleManager ClusterVersionCapability = "OperatorLifecycleManager"

	// ClusterVersionCapabilityOperatorLifecycleManagerV1 manages the Operator Lifecycle Manager (v1)
	// which itself manages the lifecycle of operators
	ClusterVersionCapabilityOperatorLifecycleManagerV1 ClusterVersionCapability = "OperatorLifecycleManagerV1"

	// ClusterVersionCapabilityCloudCredential manages credentials for cloud providers
	// in openshift cluster
	ClusterVersionCapabilityCloudCredential ClusterVersionCapability = "CloudCredential"

	// ClusterVersionCapabilityIngress manages the cluster ingress operator
	// which is responsible for running the ingress controllers (including OpenShift router).
	//
	// The following CRDs are part of the capability as well:
	// IngressController
	// DNSRecord
	// GatewayClass
	// Gateway
	// HTTPRoute
	// ReferenceGrant
	//
	// WARNING: This capability cannot be disabled on the standalone OpenShift.
	ClusterVersionCapabilityIngress ClusterVersionCapability = "Ingress"

	// ClusterVersionCapabilityCloudControllerManager manages various Cloud Controller
	// Managers deployed on top of OpenShift. They help you to work with cloud
	// provider API and embeds cloud-specific control logic.
	ClusterVersionCapabilityCloudControllerManager ClusterVersionCapability = "CloudControllerManager"
)

// KnownClusterVersionCapabilities includes all known optional, core cluster components.
var KnownClusterVersionCapabilities = []ClusterVersionCapability{
	ClusterVersionCapabilityBaremetal,
	ClusterVersionCapabilityConsole,
	ClusterVersionCapabilityInsights,
	ClusterVersionCapabilityMarketplace,
	ClusterVersionCapabilityStorage,
	ClusterVersionCapabilityOpenShiftSamples,
	ClusterVersionCapabilityCSISnapshot,
	ClusterVersionCapabilityNodeTuning,
	ClusterVersionCapabilityMachineAPI,
	ClusterVersionCapabilityBuild,
	ClusterVersionCapabilityDeploymentConfig,
	ClusterVersionCapabilityImageRegistry,
	ClusterVersionCapabilityOperatorLifecycleManager,
	ClusterVersionCapabilityOperatorLifecycleManagerV1,
	ClusterVersionCapabilityCloudCredential,
	ClusterVersionCapabilityIngress,
	ClusterVersionCapabilityCloudControllerManager,
}

// ClusterVersionCapabilitySet defines sets of cluster version capabilities.
// +kubebuilder:validation:Enum=None;v4.11;v4.12;v4.13;v4.14;v4.15;v4.16;v4.17;v4.18;vCurrent
type ClusterVersionCapabilitySet string

const (
	// ClusterVersionCapabilitySetNone is an empty set enabling
	// no optional capabilities.
	ClusterVersionCapabilitySetNone ClusterVersionCapabilitySet = "None"

	// ClusterVersionCapabilitySet4_11 is the recommended set of
	// optional capabilities to enable for the 4.11 version of
	// OpenShift.  This list will remain the same no matter which
	// version of OpenShift is installed.
	ClusterVersionCapabilitySet4_11 ClusterVersionCapabilitySet = "v4.11"

	// ClusterVersionCapabilitySet4_12 is the recommended set of
	// optional capabilities to enable for the 4.12 version of
	// OpenShift.  This list will remain the same no matter which
	// version of OpenShift is installed.
	ClusterVersionCapabilitySet4_12 ClusterVersionCapabilitySet = "v4.12"

	// ClusterVersionCapabilitySet4_13 is the recommended set of
	// optional capabilities to enable for the 4.13 version of
	// OpenShift.  This list will remain the same no matter which
	// version of OpenShift is installed.
	ClusterVersionCapabilitySet4_13 ClusterVersionCapabilitySet = "v4.13"

	// ClusterVersionCapabilitySet4_14 is the recommended set of
	// optional capabilities to enable for the 4.14 version of
	// OpenShift.  This list will remain the same no matter which
	// version of OpenShift is installed.
	ClusterVersionCapabilitySet4_14 ClusterVersionCapabilitySet = "v4.14"

	// ClusterVersionCapabilitySet4_15 is the recommended set of
	// optional capabilities to enable for the 4.15 version of
	// OpenShift.  This list will remain the same no matter which
	// version of OpenShift is installed.
	ClusterVersionCapabilitySet4_15 ClusterVersionCapabilitySet = "v4.15"

	// ClusterVersionCapabilitySet4_16 is the recommended set of
	// optional capabilities to enable for the 4.16 version of
	// OpenShift.  This list will remain the same no matter which
	// version of OpenShift is installed.
	ClusterVersionCapabilitySet4_16 ClusterVersionCapabilitySet = "v4.16"

	// ClusterVersionCapabilitySet4_17 is the recommended set of
	// optional capabilities to enable for the 4.17 version of
	// OpenShift.  This list will remain the same no matter which
	// version of OpenShift is installed.
	ClusterVersionCapabilitySet4_17 ClusterVersionCapabilitySet = "v4.17"

	// ClusterVersionCapabilitySet4_18 is the recommended set of
	// optional capabilities to enable for the 4.18 version of
	// OpenShift.  This list will remain the same no matter which
	// version of OpenShift is installed.
	ClusterVersionCapabilitySet4_18 ClusterVersionCapabilitySet = "v4.18"

	// ClusterVersionCapabilitySetCurrent is the recommended set
	// of optional capabilities to enable for the cluster's
	// current version of OpenShift.
	ClusterVersionCapabilitySetCurrent ClusterVersionCapabilitySet = "vCurrent"
)

// ClusterVersionCapabilitySets defines sets of cluster version capabilities.
var ClusterVersionCapabilitySets = map[ClusterVersionCapabilitySet][]ClusterVersionCapability{
	ClusterVersionCapabilitySetNone: {},
	ClusterVersionCapabilitySet4_11: {
		ClusterVersionCapabilityBaremetal,
		ClusterVersionCapabilityMarketplace,
		ClusterVersionCapabilityOpenShiftSamples,
		ClusterVersionCapabilityMachineAPI,
	},
	ClusterVersionCapabilitySet4_12: {
		ClusterVersionCapabilityBaremetal,
		ClusterVersionCapabilityConsole,
		ClusterVersionCapabilityInsights,
		ClusterVersionCapabilityMarketplace,
		ClusterVersionCapabilityStorage,
		ClusterVersionCapabilityOpenShiftSamples,
		ClusterVersionCapabilityCSISnapshot,
		ClusterVersionCapabilityMachineAPI,
	},
	ClusterVersionCapabilitySet4_13: {
		ClusterVersionCapabilityBaremetal,
		ClusterVersionCapabilityConsole,
		ClusterVersionCapabilityInsights,
		ClusterVersionCapabilityMarketplace,
		ClusterVersionCapabilityStorage,
		ClusterVersionCapabilityOpenShiftSamples,
		ClusterVersionCapabilityCSISnapshot,
		ClusterVersionCapabilityNodeTuning,
		ClusterVersionCapabilityMachineAPI,
	},
	ClusterVersionCapabilitySet4_14: {
		ClusterVersionCapabilityBaremetal,
		ClusterVersionCapabilityConsole,
		ClusterVersionCapabilityInsights,
		ClusterVersionCapabilityMarketplace,
		ClusterVersionCapabilityStorage,
		ClusterVersionCapabilityOpenShiftSamples,
		ClusterVersionCapabilityCSISnapshot,
		ClusterVersionCapabilityNodeTuning,
		ClusterVersionCapabilityMachineAPI,
		ClusterVersionCapabilityBuild,
		ClusterVersionCapabilityDeploymentConfig,
		ClusterVersionCapabilityImageRegistry,
	},
	ClusterVersionCapabilitySet4_15: {
		ClusterVersionCapabilityBaremetal,
		ClusterVersionCapabilityConsole,
		ClusterVersionCapabilityInsights,
		ClusterVersionCapabilityMarketplace,
		ClusterVersionCapabilityStorage,
		ClusterVersionCapabilityOpenShiftSamples,
		ClusterVersionCapabilityCSISnapshot,
		ClusterVersionCapabilityNodeTuning,
		ClusterVersionCapabilityMachineAPI,
		ClusterVersionCapabilityBuild,
		ClusterVersionCapabilityDeploymentConfig,
		ClusterVersionCapabilityImageRegistry,
		ClusterVersionCapabilityOperatorLifecycleManager,
		ClusterVersionCapabilityCloudCredential,
	},
	ClusterVersionCapabilitySet4_16: {
		ClusterVersionCapabilityBaremetal,
		ClusterVersionCapabilityConsole,
		ClusterVersionCapabilityInsights,
		ClusterVersionCapabilityMarketplace,
		ClusterVersionCapabilityStorage,
		ClusterVersionCapabilityOpenShiftSamples,
		ClusterVersionCapabilityCSISnapshot,
		ClusterVersionCapabilityNodeTuning,
		ClusterVersionCapabilityMachineAPI,
		ClusterVersionCapabilityBuild,
		ClusterVersionCapabilityDeploymentConfig,
		ClusterVersionCapabilityImageRegistry,
		ClusterVersionCapabilityOperatorLifecycleManager,
		ClusterVersionCapabilityCloudCredential,
		ClusterVersionCapabilityIngress,
		ClusterVersionCapabilityCloudControllerManager,
	},
	ClusterVersionCapabilitySet4_17: {
		ClusterVersionCapabilityBaremetal,
		ClusterVersionCapabilityConsole,
		ClusterVersionCapabilityInsights,
		ClusterVersionCapabilityMarketplace,
		ClusterVersionCapabilityStorage,
		ClusterVersionCapabilityOpenShiftSamples,
		ClusterVersionCapabilityCSISnapshot,
		ClusterVersionCapabilityNodeTuning,
		ClusterVersionCapabilityMachineAPI,
		ClusterVersionCapabilityBuild,
		ClusterVersionCapabilityDeploymentConfig,
		ClusterVersionCapabilityImageRegistry,
		ClusterVersionCapabilityOperatorLifecycleManager,
		ClusterVersionCapabilityCloudCredential,
		ClusterVersionCapabilityIngress,
		ClusterVersionCapabilityCloudControllerManager,
	},
	ClusterVersionCapabilitySet4_18: {
		ClusterVersionCapabilityBaremetal,
		ClusterVersionCapabilityConsole,
		ClusterVersionCapabilityInsights,
		ClusterVersionCapabilityMarketplace,
		ClusterVersionCapabilityStorage,
		ClusterVersionCapabilityOpenShiftSamples,
		ClusterVersionCapabilityCSISnapshot,
		ClusterVersionCapabilityNodeTuning,
		ClusterVersionCapabilityMachineAPI,
		ClusterVersionCapabilityBuild,
		ClusterVersionCapabilityDeploymentConfig,
		ClusterVersionCapabilityImageRegistry,
		ClusterVersionCapabilityOperatorLifecycleManager,
		ClusterVersionCapabilityOperatorLifecycleManagerV1,
		ClusterVersionCapabilityCloudCredential,
		ClusterVersionCapabilityIngress,
		ClusterVersionCapabilityCloudControllerManager,
	},
	ClusterVersionCapabilitySetCurrent: {
		ClusterVersionCapabilityBaremetal,
		ClusterVersionCapabilityConsole,
		ClusterVersionCapabilityInsights,
		ClusterVersionCapabilityMarketplace,
		ClusterVersionCapabilityStorage,
		ClusterVersionCapabilityOpenShiftSamples,
		ClusterVersionCapabilityCSISnapshot,
		ClusterVersionCapabilityNodeTuning,
		ClusterVersionCapabilityMachineAPI,
		ClusterVersionCapabilityBuild,
		ClusterVersionCapabilityDeploymentConfig,
		ClusterVersionCapabilityImageRegistry,
		ClusterVersionCapabilityOperatorLifecycleManager,
		ClusterVersionCapabilityOperatorLifecycleManagerV1,
		ClusterVersionCapabilityCloudCredential,
		ClusterVersionCapabilityIngress,
		ClusterVersionCapabilityCloudControllerManager,
	},
}

// ClusterVersionCapabilitiesSpec selects the managed set of
// optional, core cluster components.
// +k8s:deepcopy-gen=true
type ClusterVersionCapabilitiesSpec struct {
	// baselineCapabilitySet selects an initial set of
	// optional capabilities to enable, which can be extended via
	// additionalEnabledCapabilities.  If unset, the cluster will
	// choose a default, and the default may change over time.
	// The current default is vCurrent.
	// +optional
	BaselineCapabilitySet ClusterVersionCapabilitySet `json:"baselineCapabilitySet,omitempty"`

	// additionalEnabledCapabilities extends the set of managed
	// capabilities beyond the baseline defined in
	// baselineCapabilitySet.  The default is an empty set.
	// +listType=atomic
	// +optional
	AdditionalEnabledCapabilities []ClusterVersionCapability `json:"additionalEnabledCapabilities,omitempty"`
}

// ClusterVersionCapabilitiesStatus describes the state of optional,
// core cluster components.
// +k8s:deepcopy-gen=true
type ClusterVersionCapabilitiesStatus struct {
	// enabledCapabilities lists all the capabilities that are currently managed.
	// +listType=atomic
	// +optional
	EnabledCapabilities []ClusterVersionCapability `json:"enabledCapabilities,omitempty"`

	// knownCapabilities lists all the capabilities known to the current cluster.
	// +listType=atomic
	// +optional
	KnownCapabilities []ClusterVersionCapability `json:"knownCapabilities,omitempty"`
}

// ComponentOverride allows overriding cluster version operator's behavior
// for a component.
// +k8s:deepcopy-gen=true
type ComponentOverride struct {
	// kind indentifies which object to override.
	// +required
	Kind string `json:"kind"`
	// group identifies the API group that the kind is in.
	// +required
	Group string `json:"group"`

	// namespace is the component's namespace. If the resource is cluster
	// scoped, the namespace should be empty.
	// +required
	Namespace string `json:"namespace"`
	// name is the component's name.
	// +required
	Name string `json:"name"`

	// unmanaged controls if cluster version operator should stop managing the
	// resources in this cluster.
	// Default: false
	// +required
	Unmanaged bool `json:"unmanaged"`
}

// URL is a thin wrapper around string that ensures the string is a valid URL.
type URL string

// Update represents an administrator update request.
// +kubebuilder:validation:XValidation:rule="has(self.architecture) && has(self.image) ? (self.architecture == \"\" || self.image == \"\") : true",message="cannot set both Architecture and Image"
// +kubebuilder:validation:XValidation:rule="has(self.architecture) && self.architecture != \"\" ? self.version != \"\" : true",message="Version must be set if Architecture is set"
// +k8s:deepcopy-gen=true
type Update struct {
	// architecture is an optional field that indicates the desired
	// value of the cluster architecture. In this context cluster
	// architecture means either a single architecture or a multi
	// architecture. architecture can only be set to Multi thereby
	// only allowing updates from single to multi architecture. If
	// architecture is set, image cannot be set and version must be
	// set.
	// Valid values are 'Multi' and empty.
	//
	// +optional
	Architecture ClusterVersionArchitecture `json:"architecture"`

	// version is a semantic version identifying the update version.
	// version is required if architecture is specified.
	// If both version and image are set, the version extracted from the referenced image must match the specified version.
	//
	// +optional
	Version string `json:"version"`

	// image is a container image location that contains the update.
	// image should be used when the desired version does not exist in availableUpdates or history.
	// When image is set, architecture cannot be specified.
	// If both version and image are set, the version extracted from the referenced image must match the specified version.
	//
	// +optional
	Image string `json:"image"`

	// force allows an administrator to update to an image that has failed
	// verification or upgradeable checks. This option should only
	// be used when the authenticity of the provided image has been verified out
	// of band because the provided image will run with full administrative access
	// to the cluster. Do not use this flag with images that comes from unknown
	// or potentially malicious sources.
	//
	// +optional
	Force bool `json:"force"`
}

// Release represents an OpenShift release image and associated metadata.
// +k8s:deepcopy-gen=true
type Release struct {
	// architecture is an optional field that indicates the
	// value of the cluster architecture. In this context cluster
	// architecture means either a single architecture or a multi
	// architecture.
	// Valid values are 'Multi' and empty.
	//
	// +openshift:enable:FeatureGate=ImageStreamImportMode
	// +optional
	Architecture ClusterVersionArchitecture `json:"architecture,omitempty"`

	// version is a semantic version identifying the update version. When this
	// field is part of spec, version is optional if image is specified.
	// +required
	Version string `json:"version"`

	// image is a container image location that contains the update. When this
	// field is part of spec, image is optional if version is specified and the
	// availableUpdates field contains a matching version.
	// +required
	Image string `json:"image"`

	// url contains information about this release. This URL is set by
	// the 'url' metadata property on a release or the metadata returned by
	// the update API and should be displayed as a link in user
	// interfaces. The URL field may not be set for test or nightly
	// releases.
	// +optional
	URL URL `json:"url,omitempty"`

	// channels is the set of Cincinnati channels to which the release
	// currently belongs.
	// +listType=set
	// +optional
	Channels []string `json:"channels,omitempty"`
}

// RetrievedUpdates reports whether available updates have been retrieved from
// the upstream update server. The condition is Unknown before retrieval, False
// if the updates could not be retrieved or recently failed, or True if the
// availableUpdates field is accurate and recent.
const RetrievedUpdates ClusterStatusConditionType = "RetrievedUpdates"

// ConditionalUpdate represents an update which is recommended to some
// clusters on the version the current cluster is reconciling, but which
// may not be recommended for the current cluster.
type ConditionalUpdate struct {
	// release is the target of the update.
	// +required
	Release Release `json:"release"`

	// risks represents the range of issues associated with
	// updating to the target release. The cluster-version
	// operator will evaluate all entries, and only recommend the
	// update if there is at least one entry and all entries
	// recommend the update.
	// +kubebuilder:validation:MinItems=1
	// +patchMergeKey=name
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=name
	// +required
	Risks []ConditionalUpdateRisk `json:"risks" patchStrategy:"merge" patchMergeKey:"name"`

	// conditions represents the observations of the conditional update's
	// current status. Known types are:
	// * Recommended, for whether the update is recommended for the current cluster.
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// ConditionalUpdateRisk represents a reason and cluster-state
// for not recommending a conditional update.
// +k8s:deepcopy-gen=true
type ConditionalUpdateRisk struct {
	// url contains information about this risk.
	// +kubebuilder:validation:Format=uri
	// +kubebuilder:validation:MinLength=1
	// +required
	URL string `json:"url"`

	// name is the CamelCase reason for not recommending a
	// conditional update, in the event that matchingRules match the
	// cluster state.
	// +kubebuilder:validation:MinLength=1
	// +required
	Name string `json:"name"`

	// message provides additional information about the risk of
	// updating, in the event that matchingRules match the cluster
	// state. This is only to be consumed by humans. It may
	// contain Line Feed characters (U+000A), which should be
	// rendered as new lines.
	// +kubebuilder:validation:MinLength=1
	// +required
	Message string `json:"message"`

	// matchingRules is a slice of conditions for deciding which
	// clusters match the risk and which do not. The slice is
	// ordered by decreasing precedence. The cluster-version
	// operator will walk the slice in order, and stop after the
	// first it can successfully evaluate. If no condition can be
	// successfully evaluated, the update will not be recommended.
	// +kubebuilder:validation:MinItems=1
	// +listType=atomic
	// +required
	MatchingRules []ClusterCondition `json:"matchingRules"`
}

// ClusterCondition is a union of typed cluster conditions.  The 'type'
// property determines which of the type-specific properties are relevant.
// When evaluated on a cluster, the condition may match, not match, or
// fail to evaluate.
// +k8s:deepcopy-gen=true
type ClusterCondition struct {
	// type represents the cluster-condition type. This defines
	// the members and semantics of any additional properties.
	// +kubebuilder:validation:Enum={"Always","PromQL"}
	// +required
	Type string `json:"type"`

	// promql represents a cluster condition based on PromQL.
	// +optional
	PromQL *PromQLClusterCondition `json:"promql,omitempty"`
}

// PromQLClusterCondition represents a cluster condition based on PromQL.
type PromQLClusterCondition struct {
	// promql is a PromQL query classifying clusters. This query
	// query should return a 1 in the match case and a 0 in the
	// does-not-match case. Queries which return no time
	// series, or which return values besides 0 or 1, are
	// evaluation failures.
	// +required
	PromQL string `json:"promql"`
}

// ClusterVersionList is a list of ClusterVersion resources.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +openshift:compatibility-gen:level=1
type ClusterVersionList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []ClusterVersion `json:"items"`
}

// SignatureStore represents the URL of custom Signature Store
type SignatureStore struct {

	// url contains the upstream custom signature store URL.
	// url should be a valid absolute http/https URI of an upstream signature store as per rfc1738.
	// This must be provided and cannot be empty.
	//
	// +kubebuilder:validation:Type=string
	// +kubebuilder:validation:XValidation:rule="isURL(self)",message="url must be a valid absolute URL"
	// +required
	URL string `json:"url"`

	// ca is an optional reference to a config map by name containing the PEM-encoded CA bundle.
	// It is used as a trust anchor to validate the TLS certificate presented by the remote server.
	// The key "ca.crt" is used to locate the data.
	// If specified and the config map or expected key is not found, the signature store is not honored.
	// If the specified ca data is not valid, the signature store is not honored.
	// If empty, we fall back to the CA configured via Proxy, which is appended to the default system roots.
	// The namespace for this config map is openshift-config.
	// +optional
	CA ConfigMapNameReference `json:"ca"`
}
