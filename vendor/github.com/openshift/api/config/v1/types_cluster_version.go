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
type ClusterVersion struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec is the desired state of the cluster version - the operator will work
	// to ensure that the desired version is applied to the cluster.
	// +kubebuilder:validation:Required
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
	// +kubebuilder:validation:Required
	// +required
	ClusterID ClusterID `json:"clusterID"`

	// desiredUpdate is an optional field that indicates the desired value of
	// the cluster version. Setting this value will trigger an upgrade (if
	// the current version does not match the desired version). The set of
	// recommended update values is listed as part of available updates in
	// status, and setting values outside that range may cause the upgrade
	// to fail. You may specify the version field without setting image if
	// an update exists with that version in the availableUpdates or history.
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
	// channel is an identifier for explicitly requesting that a non-default
	// set of updates be applied to this cluster. The default channel will be
	// contain stable updates that are appropriate for production clusters.
	//
	// +optional
	Channel string `json:"channel,omitempty"`

	// capabilities configures the installation of optional, core
	// cluster components.  A null value here is identical to an
	// empty object; see the child properties for default semantics.
	// +optional
	Capabilities *ClusterVersionCapabilitiesSpec `json:"capabilities,omitempty"`

	// overrides is list of overides for components that are managed by
	// cluster version operator. Marking a component unmanaged will prevent
	// the operator from creating or updating the object.
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
	// +kubebuilder:validation:Required
	// +required
	Desired Release `json:"desired"`

	// history contains a list of the most recent versions applied to the cluster.
	// This value may be empty during cluster startup, and then will be updated
	// when a new update is being applied. The newest update is first in the
	// list and it is ordered by recency. Updates in the history have state
	// Completed if the rollout completed - if an update was failing or halfway
	// applied the state will be Partial. Only a limited amount of update history
	// is preserved.
	// +optional
	History []UpdateHistory `json:"history,omitempty"`

	// observedGeneration reports which version of the spec is being synced.
	// If this value is not equal to metadata.generation, then the desired
	// and conditions fields may represent a previous version.
	// +kubebuilder:validation:Required
	// +required
	ObservedGeneration int64 `json:"observedGeneration"`

	// versionHash is a fingerprint of the content that the cluster will be
	// updated with. It is used by the operator to avoid unnecessary work
	// and is for internal use only.
	// +kubebuilder:validation:Required
	// +required
	VersionHash string `json:"versionHash"`

	// capabilities describes the state of optional, core cluster components.
	Capabilities ClusterVersionCapabilitiesStatus `json:"capabilities"`

	// conditions provides information about the cluster version. The condition
	// "Available" is set to true if the desiredUpdate has been reached. The
	// condition "Progressing" is set to true if an update is being applied.
	// The condition "Degraded" is set to true if an update is currently blocked
	// by a temporary or permanent error. Conditions are only valid for the
	// current desiredUpdate when metadata.generation is equal to
	// status.generation.
	// +optional
	Conditions []ClusterOperatorStatusCondition `json:"conditions,omitempty"`

	// availableUpdates contains updates recommended for this
	// cluster. Updates which appear in conditionalUpdates but not in
	// availableUpdates may expose this cluster to known issues. This list
	// may be empty if no updates are recommended, if the update service
	// is unavailable, or if an invalid channel has been specified.
	// +nullable
	// +kubebuilder:validation:Required
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
	// +kubebuilder:validation:Required
	// +required
	State UpdateState `json:"state"`

	// startedTime is the time at which the update was started.
	// +kubebuilder:validation:Required
	// +required
	StartedTime metav1.Time `json:"startedTime"`

	// completionTime, if set, is when the update was fully applied. The update
	// that is currently being applied will have a null completion time.
	// Completion time will always be set for entries that are not the current
	// update (usually to the started time of the next update).
	// +kubebuilder:validation:Required
	// +required
	// +nullable
	CompletionTime *metav1.Time `json:"completionTime"`

	// version is a semantic versioning identifying the update version. If the
	// requested image does not define a version, or if a failure occurs
	// retrieving the image, this value may be empty.
	//
	// +optional
	Version string `json:"version"`

	// image is a container image location that contains the update. This value
	// is always populated.
	// +kubebuilder:validation:Required
	// +required
	Image string `json:"image"`

	// verified indicates whether the provided update was properly verified
	// before it was installed. If this is false the cluster may not be trusted.
	// Verified does not cover upgradeable checks that depend on the cluster
	// state at the time when the update target was accepted.
	// +kubebuilder:validation:Required
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

// ClusterVersionCapability enumerates optional, core cluster components.
// +kubebuilder:validation:Enum=openshift-samples;baremetal;marketplace
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
	ClusterVersionCapabilityMarketplace ClusterVersionCapability = "marketplace"
)

// KnownClusterVersionCapabilities includes all known optional, core cluster components.
var KnownClusterVersionCapabilities = []ClusterVersionCapability{
	ClusterVersionCapabilityBaremetal,
	ClusterVersionCapabilityMarketplace,
	ClusterVersionCapabilityOpenShiftSamples,
}

// ClusterVersionCapabilitySet defines sets of cluster version capabilities.
// +kubebuilder:validation:Enum=None;v4.11;vCurrent
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

	// ClusterVersionCapabilitySetCurrent is the recommended set
	// of optional capabilities to enable for the cluster's
	// current version of OpenShift.
	ClusterVersionCapabilitySetCurrent ClusterVersionCapabilitySet = "vCurrent"
)

// ClusterVersionCapabilitySets defines sets of cluster version capabilities.
var ClusterVersionCapabilitySets = map[ClusterVersionCapabilitySet][]ClusterVersionCapability{
	ClusterVersionCapabilitySetNone: {},
	ClusterVersionCapabilitySet4_11: {
		ClusterVersionCapabilityOpenShiftSamples,
		ClusterVersionCapabilityBaremetal,
		ClusterVersionCapabilityMarketplace,
	},
	ClusterVersionCapabilitySetCurrent: {
		ClusterVersionCapabilityOpenShiftSamples,
		ClusterVersionCapabilityBaremetal,
		ClusterVersionCapabilityMarketplace,
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
	// +kubebuilder:validation:Required
	// +required
	Kind string `json:"kind"`
	// group identifies the API group that the kind is in.
	// +kubebuilder:validation:Required
	// +required
	Group string `json:"group"`

	// namespace is the component's namespace. If the resource is cluster
	// scoped, the namespace should be empty.
	// +kubebuilder:validation:Required
	// +required
	Namespace string `json:"namespace"`
	// name is the component's name.
	// +kubebuilder:validation:Required
	// +required
	Name string `json:"name"`

	// unmanaged controls if cluster version operator should stop managing the
	// resources in this cluster.
	// Default: false
	// +kubebuilder:validation:Required
	// +required
	Unmanaged bool `json:"unmanaged"`
}

// URL is a thin wrapper around string that ensures the string is a valid URL.
type URL string

// Update represents an administrator update request.
// +k8s:deepcopy-gen=true
type Update struct {
	// version is a semantic versioning identifying the update version. When this
	// field is part of spec, version is optional if image is specified.
	//
	// +optional
	Version string `json:"version"`

	// image is a container image location that contains the update. When this
	// field is part of spec, image is optional if version is specified and the
	// availableUpdates field contains a matching version.
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
	// version is a semantic versioning identifying the update version. When this
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
	// +kubebuilder:validation:Required
	// +required
	Release Release `json:"release"`

	// risks represents the range of issues associated with
	// updating to the target release. The cluster-version
	// operator will evaluate all entries, and only recommend the
	// update if there is at least one entry and all entries
	// recommend the update.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinItems=1
	// +patchMergeKey=name
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=name
	// +required
	Risks []ConditionalUpdateRisk `json:"risks" patchStrategy:"merge" patchMergeKey:"name"`

	// conditions represents the observations of the conditional update's
	// current status. Known types are:
	// * Evaluating, for whether the cluster-version operator will attempt to evaluate any risks[].matchingRules.
	// * Recommended, for whether the update is recommended for the current cluster.
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,1,rep,name=conditions"`
}

// ConditionalUpdateRisk represents a reason and cluster-state
// for not recommending a conditional update.
// +k8s:deepcopy-gen=true
type ConditionalUpdateRisk struct {
	// url contains information about this risk.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Format=uri
	// +kubebuilder:validation:MinLength=1
	// +required
	URL string `json:"url"`

	// name is the CamelCase reason for not recommending a
	// conditional update, in the event that matchingRules match the
	// cluster state.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +required
	Name string `json:"name"`

	// message provides additional information about the risk of
	// updating, in the event that matchingRules match the cluster
	// state. This is only to be consumed by humans. It may
	// contain Line Feed characters (U+000A), which should be
	// rendered as new lines.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +required
	Message string `json:"message"`

	// matchingRules is a slice of conditions for deciding which
	// clusters match the risk and which do not. The slice is
	// ordered by decreasing precedence. The cluster-version
	// operator will walk the slice in order, and stop after the
	// first it can successfully evaluate. If no condition can be
	// successfully evaluated, the update will not be recommended.
	// +kubebuilder:validation:Required
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
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Enum={"Always","PromQL"}
	// +required
	Type string `json:"type"`

	// promQL represents a cluster condition based on PromQL.
	// +optional
	PromQL *PromQLClusterCondition `json:"promql,omitempty"`
}

// PromQLClusterCondition represents a cluster condition based on PromQL.
type PromQLClusterCondition struct {
	// PromQL is a PromQL query classifying clusters. This query
	// query should return a 1 in the match case and a 0 in the
	// does-not-match case. Queries which return no time
	// series, or which return values besides 0 or 1, are
	// evaluation failures.
	// +kubebuilder:validation:Required
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
	metav1.ListMeta `json:"metadata"`

	Items []ClusterVersion `json:"items"`
}
