/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
This file (together with pkg/apis/extensions/v1beta1/types.go) contain the experimental
types in kubernetes. These API objects are experimental, meaning that the
APIs may be broken at any time by the kubernetes team.

DISCLAIMER: The implementation of the experimental API group itself is
a temporary one meant as a stopgap solution until kubernetes has proper
support for multiple API groups. The transition may require changes
beyond registration differences. In other words, experimental API group
support is experimental.
*/

package extensions

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/util/intstr"
)

// describes the attributes of a scale subresource
type ScaleSpec struct {
	// desired number of instances for the scaled object.
	Replicas int `json:"replicas,omitempty"`
}

// represents the current status of a scale subresource.
type ScaleStatus struct {
	// actual number of observed instances of the scaled object.
	Replicas int `json:"replicas"`

	// label query over pods that should match the replicas count.
	// More info: http://releases.k8s.io/release-1.2/docs/user-guide/labels.md#label-selectors
	Selector *unversioned.LabelSelector `json:"selector,omitempty"`
}

// +genclient=true,noMethods=true

// represents a scaling request for a resource.
type Scale struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object metadata; More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#metadata.
	api.ObjectMeta `json:"metadata,omitempty"`

	// defines the behavior of the scale. More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#spec-and-status.
	Spec ScaleSpec `json:"spec,omitempty"`

	// current status of the scale. More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#spec-and-status. Read-only.
	Status ScaleStatus `json:"status,omitempty"`
}

// Dummy definition
type ReplicationControllerDummy struct {
	unversioned.TypeMeta `json:",inline"`
}

// SubresourceReference contains enough information to let you inspect or modify the referred subresource.
type SubresourceReference struct {
	// Kind of the referent; More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#types-kinds"
	Kind string `json:"kind,omitempty"`
	// Name of the referent; More info: http://releases.k8s.io/release-1.2/docs/user-guide/identifiers.md#names
	Name string `json:"name,omitempty"`
	// API version of the referent
	APIVersion string `json:"apiVersion,omitempty"`
	// Subresource name of the referent
	Subresource string `json:"subresource,omitempty"`
}

type CPUTargetUtilization struct {
	// fraction of the requested CPU that should be utilized/used,
	// e.g. 70 means that 70% of the requested CPU should be in use.
	TargetPercentage int `json:"targetPercentage"`
}

// Alpha-level support for Custom Metrics in HPA (as annotations).
type CustomMetricTarget struct {
	// Custom Metric name.
	Name string `json:"name"`
	// Custom Metric value (average).
	TargetValue resource.Quantity `json:"value"`
}

type CustomMetricTargetList struct {
	Items []CustomMetricTarget `json:"items"`
}

type CustomMetricCurrentStatus struct {
	// Custom Metric name.
	Name string `json:"name"`
	// Custom Metric value (average).
	CurrentValue resource.Quantity `json:"value"`
}

type CustomMetricCurrentStatusList struct {
	Items []CustomMetricCurrentStatus `json:"items"`
}

// specification of a horizontal pod autoscaler.
type HorizontalPodAutoscalerSpec struct {
	// reference to Scale subresource; horizontal pod autoscaler will learn the current resource consumption from its status,
	// and will set the desired number of pods by modifying its spec.
	ScaleRef SubresourceReference `json:"scaleRef"`
	// lower limit for the number of pods that can be set by the autoscaler, default 1.
	MinReplicas *int `json:"minReplicas,omitempty"`
	// upper limit for the number of pods that can be set by the autoscaler. It cannot be smaller than MinReplicas.
	MaxReplicas int `json:"maxReplicas"`
	// target average CPU utilization (represented as a percentage of requested CPU) over all the pods;
	// if not specified it defaults to the target CPU utilization at 80% of the requested resources.
	CPUUtilization *CPUTargetUtilization `json:"cpuUtilization,omitempty"`
}

// current status of a horizontal pod autoscaler
type HorizontalPodAutoscalerStatus struct {
	// most recent generation observed by this autoscaler.
	ObservedGeneration *int64 `json:"observedGeneration,omitempty"`

	// last time the HorizontalPodAutoscaler scaled the number of pods;
	// used by the autoscaler to control how often the number of pods is changed.
	LastScaleTime *unversioned.Time `json:"lastScaleTime,omitempty"`

	// current number of replicas of pods managed by this autoscaler.
	CurrentReplicas int `json:"currentReplicas"`

	// desired number of replicas of pods managed by this autoscaler.
	DesiredReplicas int `json:"desiredReplicas"`

	// current average CPU utilization over all pods, represented as a percentage of requested CPU,
	// e.g. 70 means that an average pod is using now 70% of its requested CPU.
	CurrentCPUUtilizationPercentage *int `json:"currentCPUUtilizationPercentage,omitempty"`
}

// +genclient=true

// configuration of a horizontal pod autoscaler.
type HorizontalPodAutoscaler struct {
	unversioned.TypeMeta `json:",inline"`
	api.ObjectMeta       `json:"metadata,omitempty"`

	// behaviour of autoscaler. More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#spec-and-status.
	Spec HorizontalPodAutoscalerSpec `json:"spec,omitempty"`

	// current information about the autoscaler.
	Status HorizontalPodAutoscalerStatus `json:"status,omitempty"`
}

// list of horizontal pod autoscaler objects.
type HorizontalPodAutoscalerList struct {
	unversioned.TypeMeta `json:",inline"`
	unversioned.ListMeta `json:"metadata,omitempty"`

	// list of horizontal pod autoscaler objects.
	Items []HorizontalPodAutoscaler `json:"items"`
}

// +genclient=true

// A ThirdPartyResource is a generic representation of a resource, it is used by add-ons and plugins to add new resource
// types to the API.  It consists of one or more Versions of the api.
type ThirdPartyResource struct {
	unversioned.TypeMeta `json:",inline"`

	// Standard object metadata
	api.ObjectMeta `json:"metadata,omitempty"`

	// Description is the description of this object.
	Description string `json:"description,omitempty"`

	// Versions are versions for this third party object
	Versions []APIVersion `json:"versions,omitempty"`
}

type ThirdPartyResourceList struct {
	unversioned.TypeMeta `json:",inline"`

	// Standard list metadata.
	unversioned.ListMeta `json:"metadata,omitempty"`

	// Items is the list of horizontal pod autoscalers.
	Items []ThirdPartyResource `json:"items"`
}

// An APIVersion represents a single concrete version of an object model.
// TODO: we should consider merge this struct with GroupVersion in unversioned.go
type APIVersion struct {
	// Name of this version (e.g. 'v1').
	Name string `json:"name,omitempty"`

	// The API group to add this object into, default 'experimental'.
	APIGroup string `json:"apiGroup,omitempty"`
}

// An internal object, used for versioned storage in etcd.  Not exposed to the end user.
type ThirdPartyResourceData struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object metadata.
	api.ObjectMeta `json:"metadata,omitempty"`

	// Data is the raw JSON data for this data.
	Data []byte `json:"data,omitempty"`
}

// +genclient=true

type Deployment struct {
	unversioned.TypeMeta `json:",inline"`
	api.ObjectMeta       `json:"metadata,omitempty"`

	// Specification of the desired behavior of the Deployment.
	Spec DeploymentSpec `json:"spec,omitempty"`

	// Most recently observed status of the Deployment.
	Status DeploymentStatus `json:"status,omitempty"`
}

type DeploymentSpec struct {
	// Number of desired pods. This is a pointer to distinguish between explicit
	// zero and not specified. Defaults to 1.
	Replicas int `json:"replicas,omitempty"`

	// Label selector for pods. Existing ReplicaSets whose pods are
	// selected by this will be the ones affected by this deployment.
	Selector *unversioned.LabelSelector `json:"selector,omitempty"`

	// Template describes the pods that will be created.
	Template api.PodTemplateSpec `json:"template"`

	// The deployment strategy to use to replace existing pods with new ones.
	Strategy DeploymentStrategy `json:"strategy,omitempty"`

	// Minimum number of seconds for which a newly created pod should be ready
	// without any of its container crashing, for it to be considered available.
	// Defaults to 0 (pod will be considered available as soon as it is ready)
	MinReadySeconds int `json:"minReadySeconds,omitempty"`

	// The number of old ReplicaSets to retain to allow rollback.
	// This is a pointer to distinguish between explicit zero and not specified.
	RevisionHistoryLimit *int `json:"revisionHistoryLimit,omitempty"`

	// Indicates that the deployment is paused and will not be processed by the
	// deployment controller.
	Paused bool `json:"paused,omitempty"`
	// The config this deployment is rolling back to. Will be cleared after rollback is done.
	RollbackTo *RollbackConfig `json:"rollbackTo,omitempty"`
}

// DeploymentRollback stores the information required to rollback a deployment.
type DeploymentRollback struct {
	unversioned.TypeMeta `json:",inline"`
	// Required: This must match the Name of a deployment.
	Name string `json:"name"`
	// The annotations to be updated to a deployment
	UpdatedAnnotations map[string]string `json:"updatedAnnotations,omitempty"`
	// The config of this deployment rollback.
	RollbackTo RollbackConfig `json:"rollbackTo"`
}

type RollbackConfig struct {
	// The revision to rollback to. If set to 0, rollbck to the last revision.
	Revision int64 `json:"revision,omitempty"`
}

const (
	// DefaultDeploymentUniqueLabelKey is the default key of the selector that is added
	// to existing RCs (and label key that is added to its pods) to prevent the existing RCs
	// to select new pods (and old pods being select by new RC).
	DefaultDeploymentUniqueLabelKey string = "pod-template-hash"
)

type DeploymentStrategy struct {
	// Type of deployment. Can be "Recreate" or "RollingUpdate". Default is RollingUpdate.
	Type DeploymentStrategyType `json:"type,omitempty"`

	// Rolling update config params. Present only if DeploymentStrategyType =
	// RollingUpdate.
	//---
	// TODO: Update this to follow our convention for oneOf, whatever we decide it
	// to be.
	RollingUpdate *RollingUpdateDeployment `json:"rollingUpdate,omitempty"`
}

type DeploymentStrategyType string

const (
	// Kill all existing pods before creating new ones.
	RecreateDeploymentStrategyType DeploymentStrategyType = "Recreate"

	// Replace the old RCs by new one using rolling update i.e gradually scale down the old RCs and scale up the new one.
	RollingUpdateDeploymentStrategyType DeploymentStrategyType = "RollingUpdate"
)

// Spec to control the desired behavior of rolling update.
type RollingUpdateDeployment struct {
	// The maximum number of pods that can be unavailable during the update.
	// Value can be an absolute number (ex: 5) or a percentage of total pods at the start of update (ex: 10%).
	// Absolute number is calculated from percentage by rounding up.
	// This can not be 0 if MaxSurge is 0.
	// By default, a fixed value of 1 is used.
	// Example: when this is set to 30%, the old RC can be scaled down by 30%
	// immediately when the rolling update starts. Once new pods are ready, old RC
	// can be scaled down further, followed by scaling up the new RC, ensuring
	// that at least 70% of original number of pods are available at all times
	// during the update.
	MaxUnavailable intstr.IntOrString `json:"maxUnavailable,omitempty"`

	// The maximum number of pods that can be scheduled above the original number of
	// pods.
	// Value can be an absolute number (ex: 5) or a percentage of total pods at
	// the start of the update (ex: 10%). This can not be 0 if MaxUnavailable is 0.
	// Absolute number is calculated from percentage by rounding up.
	// By default, a value of 1 is used.
	// Example: when this is set to 30%, the new RC can be scaled up by 30%
	// immediately when the rolling update starts. Once old pods have been killed,
	// new RC can be scaled up further, ensuring that total number of pods running
	// at any time during the update is atmost 130% of original pods.
	MaxSurge intstr.IntOrString `json:"maxSurge,omitempty"`
}

type DeploymentStatus struct {
	// The generation observed by the deployment controller.
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Total number of non-terminated pods targeted by this deployment (their labels match the selector).
	Replicas int `json:"replicas,omitempty"`

	// Total number of non-terminated pods targeted by this deployment that have the desired template spec.
	UpdatedReplicas int `json:"updatedReplicas,omitempty"`

	// Total number of available pods (ready for at least minReadySeconds) targeted by this deployment.
	AvailableReplicas int `json:"availableReplicas,omitempty"`

	// Total number of unavailable pods targeted by this deployment.
	UnavailableReplicas int `json:"unavailableReplicas,omitempty"`
}

type DeploymentList struct {
	unversioned.TypeMeta `json:",inline"`
	unversioned.ListMeta `json:"metadata,omitempty"`

	// Items is the list of deployments.
	Items []Deployment `json:"items"`
}

// TODO(madhusudancs): Uncomment while implementing DaemonSet updates.
/* Commenting out for v1.2. We are planning to bring these types back with a more robust DaemonSet update implementation in v1.3, hence not deleting but just commenting the types out.
type DaemonSetUpdateStrategy struct {
	// Type of daemon set update. Only "RollingUpdate" is supported at this time. Default is RollingUpdate.
	Type DaemonSetUpdateStrategyType `json:"type,omitempty"`

	// Rolling update config params. Present only if DaemonSetUpdateStrategy =
	// RollingUpdate.
	//---
	// TODO: Update this to follow our convention for oneOf, whatever we decide it
	// to be. Same as DeploymentStrategy.RollingUpdate.
	RollingUpdate *RollingUpdateDaemonSet `json:"rollingUpdate,omitempty"`
}

type DaemonSetUpdateStrategyType string

const (
	// Replace the old daemons by new ones using rolling update i.e replace them on each node one after the other.
	RollingUpdateDaemonSetStrategyType DaemonSetUpdateStrategyType = "RollingUpdate"
)

// Spec to control the desired behavior of daemon set rolling update.
type RollingUpdateDaemonSet struct {
	// The maximum number of DaemonSet pods that can be unavailable during the
	// update. Value can be an absolute number (ex: 5) or a percentage of total
	// number of DaemonSet pods at the start of the update (ex: 10%). Absolute
	// number is calculated from percentage by rounding up.
	// This cannot be 0.
	// Default value is 1.
	// Example: when this is set to 30%, 30% of the currently running DaemonSet
	// pods can be stopped for an update at any given time. The update starts
	// by stopping at most 30% of the currently running DaemonSet pods and then
	// brings up new DaemonSet pods in their place. Once the new pods are ready,
	// it then proceeds onto other DaemonSet pods, thus ensuring that at least
	// 70% of original number of DaemonSet pods are available at all times
	// during the update.
	MaxUnavailable intstr.IntOrString `json:"maxUnavailable,omitempty"`

	// Minimum number of seconds for which a newly created DaemonSet pod should
	// be ready without any of its container crashing, for it to be considered
	// available. Defaults to 0 (pod will be considered available as soon as it
	// is ready).
	MinReadySeconds int `json:"minReadySeconds,omitempty"`
}
*/

// DaemonSetSpec is the specification of a daemon set.
type DaemonSetSpec struct {
	// Selector is a label query over pods that are managed by the daemon set.
	// Must match in order to be controlled.
	// If empty, defaulted to labels on Pod template.
	// More info: http://releases.k8s.io/release-1.2/docs/user-guide/labels.md#label-selectors
	Selector *unversioned.LabelSelector `json:"selector,omitempty"`

	// Template is the object that describes the pod that will be created.
	// The DaemonSet will create exactly one copy of this pod on every node
	// that matches the template's node selector (or on every node if no node
	// selector is specified).
	// More info: http://releases.k8s.io/release-1.2/docs/user-guide/replication-controller.md#pod-template
	Template api.PodTemplateSpec `json:"template"`

	// TODO(madhusudancs): Uncomment while implementing DaemonSet updates.
	/* Commenting out for v1.2. We are planning to bring these fields back with a more robust DaemonSet update implementation in v1.3, hence not deleting but just commenting these fields out.
	// Update strategy to replace existing DaemonSet pods with new pods.
	UpdateStrategy DaemonSetUpdateStrategy `json:"updateStrategy,omitempty"`

	// Label key that is added to DaemonSet pods to distinguish between old and
	// new pod templates during DaemonSet update.
	// Users can set this to an empty string to indicate that the system should
	// not add any label. If unspecified, system uses
	// DefaultDaemonSetUniqueLabelKey("daemonset.kubernetes.io/podTemplateHash").
	// Value of this key is hash of DaemonSetSpec.PodTemplateSpec.
	// No label is added if this is set to empty string.
	UniqueLabelKey string `json:"uniqueLabelKey,omitempty"`
	*/
}

const (
	// DefaultDaemonSetUniqueLabelKey is the default key of the labels that is added
	// to daemon set pods to distinguish between old and new pod templates during
	// DaemonSet update. See DaemonSetSpec's UniqueLabelKey field for more information.
	DefaultDaemonSetUniqueLabelKey string = "daemonset.kubernetes.io/podTemplateHash"
)

// DaemonSetStatus represents the current status of a daemon set.
type DaemonSetStatus struct {
	// CurrentNumberScheduled is the number of nodes that are running at least 1
	// daemon pod and are supposed to run the daemon pod.
	CurrentNumberScheduled int `json:"currentNumberScheduled"`

	// NumberMisscheduled is the number of nodes that are running the daemon pod, but are
	// not supposed to run the daemon pod.
	NumberMisscheduled int `json:"numberMisscheduled"`

	// DesiredNumberScheduled is the total number of nodes that should be running the daemon
	// pod (including nodes correctly running the daemon pod).
	DesiredNumberScheduled int `json:"desiredNumberScheduled"`
}

// +genclient=true

// DaemonSet represents the configuration of a daemon set.
type DaemonSet struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#metadata
	api.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the desired behavior of this daemon set.
	// More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#spec-and-status
	Spec DaemonSetSpec `json:"spec,omitempty"`

	// Status is the current status of this daemon set. This data may be
	// out of date by some window of time.
	// Populated by the system.
	// Read-only.
	// More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#spec-and-status
	Status DaemonSetStatus `json:"status,omitempty"`
}

// DaemonSetList is a collection of daemon sets.
type DaemonSetList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#metadata
	unversioned.ListMeta `json:"metadata,omitempty"`

	// Items is a list of daemon sets.
	Items []DaemonSet `json:"items"`
}

type ThirdPartyResourceDataList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata
	// More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#metadata
	unversioned.ListMeta `json:"metadata,omitempty"`
	// Items is a list of third party objects
	Items []ThirdPartyResourceData `json:"items"`
}

// +genclient=true

// Job represents the configuration of a single job.
type Job struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#metadata
	api.ObjectMeta `json:"metadata,omitempty"`

	// Spec is a structure defining the expected behavior of a job.
	// More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#spec-and-status
	Spec JobSpec `json:"spec,omitempty"`

	// Status is a structure describing current status of a job.
	// More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#spec-and-status
	Status JobStatus `json:"status,omitempty"`
}

// JobList is a collection of jobs.
type JobList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata
	// More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#metadata
	unversioned.ListMeta `json:"metadata,omitempty"`

	// Items is the list of Job.
	Items []Job `json:"items"`
}

// JobSpec describes how the job execution will look like.
type JobSpec struct {

	// Parallelism specifies the maximum desired number of pods the job should
	// run at any given time. The actual number of pods running in steady state will
	// be less than this number when ((.spec.completions - .status.successful) < .spec.parallelism),
	// i.e. when the work left to do is less than max parallelism.
	Parallelism *int `json:"parallelism,omitempty"`

	// Completions specifies the desired number of successfully finished pods the
	// job should be run with.  Setting to nil means that the success of any
	// pod signals the success of all pods, and allows parallelism to have any positive
	// value.  Setting to 1 means that parallelism is limited to 1 and the success of that
	// pod signals the success of the job.
	Completions *int `json:"completions,omitempty"`

	// Optional duration in seconds relative to the startTime that the job may be active
	// before the system tries to terminate it; value must be positive integer
	ActiveDeadlineSeconds *int64 `json:"activeDeadlineSeconds,omitempty"`

	// Selector is a label query over pods that should match the pod count.
	// Normally, the system sets this field for you.
	Selector *unversioned.LabelSelector `json:"selector,omitempty"`

	// ManualSelector controls generation of pod labels and pod selectors.
	// Leave `manualSelector` unset unless you are certain what you are doing.
	// When false or unset, the system pick labels unique to this job
	// and appends those labels to the pod template.  When true,
	// the user is responsible for picking unique labels and specifying
	// the selector.  Failure to pick a unique label may cause this
	// and other jobs to not function correctly.  However, You may see
	// `manualSelector=true` in jobs that were created with the old `extensions/v1beta1`
	// API.
	ManualSelector *bool `json:"manualSelector,omitempty"`

	// Template is the object that describes the pod that will be created when
	// executing a job.
	Template api.PodTemplateSpec `json:"template"`
}

// JobStatus represents the current state of a Job.
type JobStatus struct {

	// Conditions represent the latest available observations of an object's current state.
	Conditions []JobCondition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`

	// StartTime represents time when the job was acknowledged by the Job Manager.
	// It is not guaranteed to be set in happens-before order across separate operations.
	// It is represented in RFC3339 form and is in UTC.
	StartTime *unversioned.Time `json:"startTime,omitempty"`

	// CompletionTime represents time when the job was completed. It is not guaranteed to
	// be set in happens-before order across separate operations.
	// It is represented in RFC3339 form and is in UTC.
	CompletionTime *unversioned.Time `json:"completionTime,omitempty"`

	// Active is the number of actively running pods.
	Active int `json:"active,omitempty"`

	// Succeeded is the number of pods which reached Phase Succeeded.
	Succeeded int `json:"succeeded,omitempty"`

	// Failed is the number of pods which reached Phase Failed.
	Failed int `json:"failed,omitempty"`
}

type JobConditionType string

// These are valid conditions of a job.
const (
	// JobComplete means the job has completed its execution.
	JobComplete JobConditionType = "Complete"
	// JobFailed means the job has failed its execution.
	JobFailed JobConditionType = "Failed"
)

// JobCondition describes current state of a job.
type JobCondition struct {
	// Type of job condition, Complete or Failed.
	Type JobConditionType `json:"type"`
	// Status of the condition, one of True, False, Unknown.
	Status api.ConditionStatus `json:"status"`
	// Last time the condition was checked.
	LastProbeTime unversioned.Time `json:"lastProbeTime,omitempty"`
	// Last time the condition transit from one status to another.
	LastTransitionTime unversioned.Time `json:"lastTransitionTime,omitempty"`
	// (brief) reason for the condition's last transition.
	Reason string `json:"reason,omitempty"`
	// Human readable message indicating details about last transition.
	Message string `json:"message,omitempty"`
}

// +genclient=true

// Ingress is a collection of rules that allow inbound connections to reach the
// endpoints defined by a backend. An Ingress can be configured to give services
// externally-reachable urls, load balance traffic, terminate SSL, offer name
// based virtual hosting etc.
type Ingress struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#metadata
	api.ObjectMeta `json:"metadata,omitempty"`

	// Spec is the desired state of the Ingress.
	// More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#spec-and-status
	Spec IngressSpec `json:"spec,omitempty"`

	// Status is the current state of the Ingress.
	// More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#spec-and-status
	Status IngressStatus `json:"status,omitempty"`
}

// IngressList is a collection of Ingress.
type IngressList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#metadata
	unversioned.ListMeta `json:"metadata,omitempty"`

	// Items is the list of Ingress.
	Items []Ingress `json:"items"`
}

// IngressSpec describes the Ingress the user wishes to exist.
type IngressSpec struct {
	// A default backend capable of servicing requests that don't match any
	// rule. At least one of 'backend' or 'rules' must be specified. This field
	// is optional to allow the loadbalancer controller or defaulting logic to
	// specify a global default.
	Backend *IngressBackend `json:"backend,omitempty"`

	// TLS is the TLS configuration. Currently the Ingress only supports a single TLS
	// port, 443, and assumes TLS termination. If multiple members of this
	// list specify different hosts, they will be multiplexed on the same
	// port according to the hostname specified through the SNI TLS extension.
	TLS []IngressTLS `json:"tls,omitempty"`

	// A list of host rules used to configure the Ingress. If unspecified, or
	// no rule matches, all traffic is sent to the default backend.
	Rules []IngressRule `json:"rules,omitempty"`
	// TODO: Add the ability to specify load-balancer IP through claims
}

// IngressTLS describes the transport layer security associated with an Ingress.
type IngressTLS struct {
	// Hosts are a list of hosts included in the TLS certificate. The values in
	// this list must match the name/s used in the tlsSecret. Defaults to the
	// wildcard host setting for the loadbalancer controller fulfilling this
	// Ingress, if left unspecified.
	Hosts []string `json:"hosts,omitempty"`
	// SecretName is the name of the secret used to terminate SSL traffic on 443.
	// Field is left optional to allow SSL routing based on SNI hostname alone.
	// If the SNI host in a listener conflicts with the "Host" header field used
	// by an IngressRule, the SNI host is used for termination and value of the
	// Host header is used for routing.
	SecretName string `json:"secretName,omitempty"`
	// TODO: Consider specifying different modes of termination, protocols etc.
}

// IngressStatus describe the current state of the Ingress.
type IngressStatus struct {
	// LoadBalancer contains the current status of the load-balancer.
	LoadBalancer api.LoadBalancerStatus `json:"loadBalancer,omitempty"`
}

// IngressRule represents the rules mapping the paths under a specified host to
// the related backend services. Incoming requests are first evaluated for a host
// match, then routed to the backend associated with the matching IngressRuleValue.
type IngressRule struct {
	// Host is the fully qualified domain name of a network host, as defined
	// by RFC 3986. Note the following deviations from the "host" part of the
	// URI as defined in the RFC:
	// 1. IPs are not allowed. Currently an IngressRuleValue can only apply to the
	//	  IP in the Spec of the parent Ingress.
	// 2. The `:` delimiter is not respected because ports are not allowed.
	//	  Currently the port of an Ingress is implicitly :80 for http and
	//	  :443 for https.
	// Both these may change in the future.
	// Incoming requests are matched against the host before the IngressRuleValue.
	// If the host is unspecified, the Ingress routes all traffic based on the
	// specified IngressRuleValue.
	Host string `json:"host,omitempty"`
	// IngressRuleValue represents a rule to route requests for this IngressRule.
	// If unspecified, the rule defaults to a http catch-all. Whether that sends
	// just traffic matching the host to the default backend or all traffic to the
	// default backend, is left to the controller fulfilling the Ingress. Http is
	// currently the only supported IngressRuleValue.
	IngressRuleValue `json:",inline,omitempty"`
}

// IngressRuleValue represents a rule to apply against incoming requests. If the
// rule is satisfied, the request is routed to the specified backend. Currently
// mixing different types of rules in a single Ingress is disallowed, so exactly
// one of the following must be set.
type IngressRuleValue struct {
	//TODO:
	// 1. Consider renaming this resource and the associated rules so they
	// aren't tied to Ingress. They can be used to route intra-cluster traffic.
	// 2. Consider adding fields for ingress-type specific global options
	// usable by a loadbalancer, like http keep-alive.

	HTTP *HTTPIngressRuleValue `json:"http,omitempty"`
}

// HTTPIngressRuleValue is a list of http selectors pointing to backends.
// In the example: http://<host>/<path>?<searchpart> -> backend where
// where parts of the url correspond to RFC 3986, this resource will be used
// to match against everything after the last '/' and before the first '?'
// or '#'.
type HTTPIngressRuleValue struct {
	// A collection of paths that map requests to backends.
	Paths []HTTPIngressPath `json:"paths"`
	// TODO: Consider adding fields for ingress-type specific global
	// options usable by a loadbalancer, like http keep-alive.
}

// HTTPIngressPath associates a path regex with a backend. Incoming urls matching
// the path are forwarded to the backend.
type HTTPIngressPath struct {
	// Path is a extended POSIX regex as defined by IEEE Std 1003.1,
	// (i.e this follows the egrep/unix syntax, not the perl syntax)
	// matched against the path of an incoming request. Currently it can
	// contain characters disallowed from the conventional "path"
	// part of a URL as defined by RFC 3986. Paths must begin with
	// a '/'. If unspecified, the path defaults to a catch all sending
	// traffic to the backend.
	Path string `json:"path,omitempty"`

	// Backend defines the referenced service endpoint to which the traffic
	// will be forwarded to.
	Backend IngressBackend `json:"backend"`
}

// IngressBackend describes all endpoints for a given service and port.
type IngressBackend struct {
	// Specifies the name of the referenced service.
	ServiceName string `json:"serviceName"`

	// Specifies the port of the referenced service.
	ServicePort intstr.IntOrString `json:"servicePort"`
}

// +genclient=true

// ReplicaSet represents the configuration of a replica set.
type ReplicaSet struct {
	unversioned.TypeMeta `json:",inline"`
	api.ObjectMeta       `json:"metadata,omitempty"`

	// Spec defines the desired behavior of this ReplicaSet.
	Spec ReplicaSetSpec `json:"spec,omitempty"`

	// Status is the current status of this ReplicaSet. This data may be
	// out of date by some window of time.
	Status ReplicaSetStatus `json:"status,omitempty"`
}

// ReplicaSetList is a collection of ReplicaSets.
type ReplicaSetList struct {
	unversioned.TypeMeta `json:",inline"`
	unversioned.ListMeta `json:"metadata,omitempty"`

	Items []ReplicaSet `json:"items"`
}

// ReplicaSetSpec is the specification of a ReplicaSet.
// As the internal representation of a ReplicaSet, it must have
// a Template set.
type ReplicaSetSpec struct {
	// Replicas is the number of desired replicas.
	Replicas int `json:"replicas"`

	// Selector is a label query over pods that should match the replica count.
	// Must match in order to be controlled.
	// If empty, defaulted to labels on pod template.
	// More info: http://releases.k8s.io/release-1.2/docs/user-guide/labels.md#label-selectors
	Selector *unversioned.LabelSelector `json:"selector,omitempty"`

	// Template is the object that describes the pod that will be created if
	// insufficient replicas are detected.
	Template api.PodTemplateSpec `json:"template,omitempty"`
}

// ReplicaSetStatus represents the current status of a ReplicaSet.
type ReplicaSetStatus struct {
	// Replicas is the number of actual replicas.
	Replicas int `json:"replicas"`

	// The number of pods that have labels matching the labels of the pod template of the replicaset.
	FullyLabeledReplicas int `json:"fullyLabeledReplicas,omitempty"`

	// ObservedGeneration is the most recent generation observed by the controller.
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
}

// PodSecurityPolicy governs the ability to make requests that affect the SecurityContext
// that will be applied to a pod and container.
type PodSecurityPolicy struct {
	unversioned.TypeMeta `json:",inline"`
	api.ObjectMeta       `json:"metadata,omitempty"`

	// Spec defines the policy enforced.
	Spec PodSecurityPolicySpec `json:"spec,omitempty"`
}

// PodSecurityPolicySpec defines the policy enforced.
type PodSecurityPolicySpec struct {
	// Privileged determines if a pod can request to be run as privileged.
	Privileged bool `json:"privileged,omitempty"`
	// Capabilities is a list of capabilities that can be added.
	Capabilities []api.Capability `json:"capabilities,omitempty"`
	// Volumes is a white list of allowed volume plugins.  Empty indicates that all plugins
	// may be used.
	Volumes []FSType `json:"volumes,omitempty"`
	// HostNetwork determines if the policy allows the use of HostNetwork in the pod spec.
	HostNetwork bool `json:"hostNetwork,omitempty"`
	// HostPorts determines which host port ranges are allowed to be exposed.
	HostPorts []HostPortRange `json:"hostPorts,omitempty"`
	// HostPID determines if the policy allows the use of HostPID in the pod spec.
	HostPID bool `json:"hostPID,omitempty"`
	// HostIPC determines if the policy allows the use of HostIPC in the pod spec.
	HostIPC bool `json:"hostIPC,omitempty"`
	// SELinux is the strategy that will dictate the allowable labels that may be set.
	SELinux SELinuxStrategyOptions `json:"seLinux,omitempty"`
	// RunAsUser is the strategy that will dictate the allowable RunAsUser values that may be set.
	RunAsUser RunAsUserStrategyOptions `json:"runAsUser,omitempty"`
}

// HostPortRange defines a range of host ports that will be enabled by a policy
// for pods to use.  It requires both the start and end to be defined.
type HostPortRange struct {
	// Min is the start of the range, inclusive.
	Min int `json:"min"`
	// Max is the end of the range, inclusive.
	Max int `json:"max"`
}

// FSType gives strong typing to different file systems that are used by volumes.
type FSType string

var (
	HostPath              FSType = "hostPath"
	EmptyDir              FSType = "emptyDir"
	GCEPersistentDisk     FSType = "gcePersistentDisk"
	AWSElasticBlockStore  FSType = "awsElasticBlockStore"
	GitRepo               FSType = "gitRepo"
	Secret                FSType = "secret"
	NFS                   FSType = "nfs"
	ISCSI                 FSType = "iscsi"
	Glusterfs             FSType = "glusterfs"
	PersistentVolumeClaim FSType = "persistentVolumeClaim"
	RBD                   FSType = "rbd"
	Cinder                FSType = "cinder"
	CephFS                FSType = "cephFS"
	DownwardAPI           FSType = "downwardAPI"
	FC                    FSType = "fc"
)

// SELinuxStrategyOptions defines the strategy type and any options used to create the strategy.
type SELinuxStrategyOptions struct {
	// Rule is the strategy that will dictate the allowable labels that may be set.
	Rule SELinuxStrategy `json:"rule"`
	// seLinuxOptions required to run as; required for MustRunAs
	// More info: http://releases.k8s.io/release-1.2/docs/design/security_context.md#security-context
	SELinuxOptions *api.SELinuxOptions `json:"seLinuxOptions,omitempty"`
}

// SELinuxStrategy denotes strategy types for generating SELinux options for a
// Security.
type SELinuxStrategy string

const (
	// container must have SELinux labels of X applied.
	SELinuxStrategyMustRunAs SELinuxStrategy = "MustRunAs"
	// container may make requests for any SELinux context labels.
	SELinuxStrategyRunAsAny SELinuxStrategy = "RunAsAny"
)

// RunAsUserStrategyOptions defines the strategy type and any options used to create the strategy.
type RunAsUserStrategyOptions struct {
	// Rule is the strategy that will dictate the allowable RunAsUser values that may be set.
	Rule RunAsUserStrategy `json:"rule"`
	// Ranges are the allowed ranges of uids that may be used.
	Ranges []IDRange `json:"ranges,omitempty"`
}

// IDRange provides a min/max of an allowed range of IDs.
type IDRange struct {
	// Min is the start of the range, inclusive.
	Min int64 `json:"min"`
	// Max is the end of the range, inclusive.
	Max int64 `json:"max"`
}

// RunAsUserStrategy denotes strategy types for generating RunAsUser values for a
// SecurityContext.
type RunAsUserStrategy string

const (
	// container must run as a particular uid.
	RunAsUserStrategyMustRunAs RunAsUserStrategy = "MustRunAs"
	// container must run as a non-root uid
	RunAsUserStrategyMustRunAsNonRoot RunAsUserStrategy = "MustRunAsNonRoot"
	// container may make requests for any uid.
	RunAsUserStrategyRunAsAny RunAsUserStrategy = "RunAsAny"
)

// PodSecurityPolicyList is a list of PodSecurityPolicy objects.
type PodSecurityPolicyList struct {
	unversioned.TypeMeta `json:",inline"`
	unversioned.ListMeta `json:"metadata,omitempty"`

	Items []PodSecurityPolicy `json:"items"`
}
