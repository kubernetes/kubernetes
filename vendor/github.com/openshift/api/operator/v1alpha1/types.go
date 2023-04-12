package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	configv1 "github.com/openshift/api/config/v1"
)

type ManagementState string

const (
	// Managed means that the operator is actively managing its resources and trying to keep the component active
	Managed ManagementState = "Managed"
	// Unmanaged means that the operator is not taking any action related to the component
	Unmanaged ManagementState = "Unmanaged"
	// Removed means that the operator is actively managing its resources and trying to remove all traces of the component
	Removed ManagementState = "Removed"
)

// OperatorSpec contains common fields for an operator to need.  It is intended to be anonymous included
// inside of the Spec struct for you particular operator.
type OperatorSpec struct {
	// managementState indicates whether and how the operator should manage the component
	ManagementState ManagementState `json:"managementState"`

	// imagePullSpec is the image to use for the component.
	ImagePullSpec string `json:"imagePullSpec"`

	// imagePullPolicy specifies the image pull policy. One of Always, Never, IfNotPresent. Defaults to Always if :latest tag is specified,
	// or IfNotPresent otherwise.
	ImagePullPolicy string `json:"imagePullPolicy"`

	// version is the desired state in major.minor.micro-patch.  Usually patch is ignored.
	Version string `json:"version"`

	// logging contains glog parameters for the component pods.  It's always a command line arg for the moment
	Logging LoggingConfig `json:"logging,omitempty"`
}

// LoggingConfig holds information about configuring logging
type LoggingConfig struct {
	// level is passed to glog.
	Level int64 `json:"level"`

	// vmodule is passed to glog.
	Vmodule string `json:"vmodule"`
}

type ConditionStatus string

const (
	ConditionTrue    ConditionStatus = "True"
	ConditionFalse   ConditionStatus = "False"
	ConditionUnknown ConditionStatus = "Unknown"

	// these conditions match the conditions for the ClusterOperator type.
	OperatorStatusTypeAvailable   = "Available"
	OperatorStatusTypeProgressing = "Progressing"
	OperatorStatusTypeFailing     = "Failing"

	OperatorStatusTypeMigrating = "Migrating"
	// TODO this is going to be removed
	OperatorStatusTypeSyncSuccessful = "SyncSuccessful"
)

// OperatorCondition is just the standard condition fields.
type OperatorCondition struct {
	Type               string          `json:"type"`
	Status             ConditionStatus `json:"status"`
	LastTransitionTime metav1.Time     `json:"lastTransitionTime,omitempty"`
	Reason             string          `json:"reason,omitempty"`
	Message            string          `json:"message,omitempty"`
}

// VersionAvailability gives information about the synchronization and operational status of a particular version of the component
type VersionAvailability struct {
	// version is the level this availability applies to
	Version string `json:"version"`
	// updatedReplicas indicates how many replicas are at the desired state
	UpdatedReplicas int32 `json:"updatedReplicas"`
	// readyReplicas indicates how many replicas are ready and at the desired state
	ReadyReplicas int32 `json:"readyReplicas"`
	// errors indicates what failures are associated with the operator trying to manage this version
	Errors []string `json:"errors"`
	// generations allows an operator to track what the generation of "important" resources was the last time we updated them
	Generations []GenerationHistory `json:"generations"`
}

// GenerationHistory keeps track of the generation for a given resource so that decisions about forced updated can be made.
type GenerationHistory struct {
	// group is the group of the thing you're tracking
	Group string `json:"group"`
	// resource is the resource type of the thing you're tracking
	Resource string `json:"resource"`
	// namespace is where the thing you're tracking is
	Namespace string `json:"namespace"`
	// name is the name of the thing you're tracking
	Name string `json:"name"`
	// lastGeneration is the last generation of the workload controller involved
	LastGeneration int64 `json:"lastGeneration"`
}

// OperatorStatus contains common fields for an operator to need.  It is intended to be anonymous included
// inside of the Status struct for you particular operator.
type OperatorStatus struct {
	// observedGeneration is the last generation change you've dealt with
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// conditions is a list of conditions and their status
	Conditions []OperatorCondition `json:"conditions,omitempty"`

	// state indicates what the operator has observed to be its current operational status.
	State ManagementState `json:"state,omitempty"`
	// taskSummary is a high level summary of what the controller is currently attempting to do.  It is high-level, human-readable
	// and not guaranteed in any way. (I needed this for debugging and realized it made a great summary).
	TaskSummary string `json:"taskSummary,omitempty"`

	// currentVersionAvailability is availability information for the current version.  If it is unmanged or removed, this doesn't exist.
	CurrentAvailability *VersionAvailability `json:"currentVersionAvailability,omitempty"`
	// targetVersionAvailability is availability information for the target version if we are migrating
	TargetAvailability *VersionAvailability `json:"targetVersionAvailability,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// GenericOperatorConfig provides information to configure an operator
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:internal
type GenericOperatorConfig struct {
	metav1.TypeMeta `json:",inline"`

	// ServingInfo is the HTTP serving information for the controller's endpoints
	ServingInfo configv1.HTTPServingInfo `json:"servingInfo,omitempty"`

	// leaderElection provides information to elect a leader. Only override this if you have a specific need
	LeaderElection configv1.LeaderElection `json:"leaderElection,omitempty"`

	// authentication allows configuration of authentication for the endpoints
	Authentication DelegatedAuthentication `json:"authentication,omitempty"`
	// authorization allows configuration of authentication for the endpoints
	Authorization DelegatedAuthorization `json:"authorization,omitempty"`
}

// DelegatedAuthentication allows authentication to be disabled.
type DelegatedAuthentication struct {
	// disabled indicates that authentication should be disabled.  By default it will use delegated authentication.
	Disabled bool `json:"disabled,omitempty"`
}

// DelegatedAuthorization allows authorization to be disabled.
type DelegatedAuthorization struct {
	// disabled indicates that authorization should be disabled.  By default it will use delegated authorization.
	Disabled bool `json:"disabled,omitempty"`
}

// StaticPodOperatorStatus is status for controllers that manage static pods.  There are different needs because individual
// node status must be tracked.
type StaticPodOperatorStatus struct {
	OperatorStatus `json:",inline"`

	// latestAvailableDeploymentGeneration is the deploymentID of the most recent deployment
	LatestAvailableDeploymentGeneration int32 `json:"latestAvailableDeploymentGeneration"`

	// nodeStatuses track the deployment values and errors across individual nodes
	NodeStatuses []NodeStatus `json:"nodeStatuses"`
}

// NodeStatus provides information about the current state of a particular node managed by this operator.
type NodeStatus struct {
	// nodeName is the name of the node
	NodeName string `json:"nodeName"`

	// currentDeploymentGeneration is the generation of the most recently successful deployment
	CurrentDeploymentGeneration int32 `json:"currentDeploymentGeneration"`
	// targetDeploymentGeneration is the generation of the deployment we're trying to apply
	TargetDeploymentGeneration int32 `json:"targetDeploymentGeneration"`
	// lastFailedDeploymentGeneration is the generation of the deployment we tried and failed to deploy.
	LastFailedDeploymentGeneration int32 `json:"lastFailedDeploymentGeneration"`

	// lastFailedDeploymentGenerationErrors is a list of the errors during the failed deployment referenced in lastFailedDeploymentGeneration
	LastFailedDeploymentErrors []string `json:"lastFailedDeploymentErrors"`
}
