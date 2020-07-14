package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ClusterVersion is the configuration for the ClusterVersionOperator. This is where
// parameters related to automatic updates can be set.
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
	Desired Update `json:"desired"`

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

	// conditions provides information about the cluster version. The condition
	// "Available" is set to true if the desiredUpdate has been reached. The
	// condition "Progressing" is set to true if an update is being applied.
	// The condition "Degraded" is set to true if an update is currently blocked
	// by a temporary or permanent error. Conditions are only valid for the
	// current desiredUpdate when metadata.generation is equal to
	// status.generation.
	// +optional
	Conditions []ClusterOperatorStatusCondition `json:"conditions,omitempty"`

	// availableUpdates contains the list of updates that are appropriate
	// for this cluster. This list may be empty if no updates are recommended,
	// if the update service is unavailable, or if an invalid channel has
	// been specified.
	// +nullable
	// +kubebuilder:validation:Required
	// +required
	AvailableUpdates []Update `json:"availableUpdates"`
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
	// +kubebuilder:validation:Required
	// +required
	Verified bool `json:"verified"`
}

// ClusterID is string RFC4122 uuid.
type ClusterID string

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

// Update represents a release of the ClusterVersionOperator, referenced by the
// Image member.
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
	// verification, does not appear in the availableUpdates list, or otherwise
	// would be blocked by normal protections on update. This option should only
	// be used when the authenticity of the provided image has been verified out
	// of band because the provided image will run with full administrative access
	// to the cluster. Do not use this flag with images that comes from unknown
	// or potentially malicious sources.
	//
	// This flag does not override other forms of consistency checking that are
	// required before a new update is deployed.
	//
	// +optional
	Force bool `json:"force"`
}

// RetrievedUpdates reports whether available updates have been retrieved from
// the upstream update server. The condition is Unknown before retrieval, False
// if the updates could not be retrieved or recently failed, or True if the
// availableUpdates field is accurate and recent.
const RetrievedUpdates ClusterStatusConditionType = "RetrievedUpdates"

// ClusterVersionList is a list of ClusterVersion resources.
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ClusterVersionList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []ClusterVersion `json:"items"`
}
