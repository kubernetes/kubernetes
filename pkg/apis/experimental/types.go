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
This file (together with pkg/apis/experimental/v1/types.go) contain the experimental
types in kubernetes. These API objects are experimental, meaning that the
APIs may be broken at any time by the kubernetes team.

DISCLAIMER: The implementation of the experimental API group itself is
a temporary one meant as a stopgap solution until kubernetes has proper
support for multiple API groups. The transition may require changes
beyond registration differences. In other words, experimental API group
support is experimental.
*/

package experimental

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/util"
)

// ScaleSpec describes the attributes a Scale subresource
type ScaleSpec struct {
	// Replicas is the number of desired replicas. More info: http://releases.k8s.io/HEAD/docs/user-guide/replication-controller.md#what-is-a-replication-controller"
	Replicas int `json:"replicas,omitempty"`
}

// ScaleStatus represents the current status of a Scale subresource.
type ScaleStatus struct {
	// Replicas is the number of actual replicas. More info: http://releases.k8s.io/HEAD/docs/user-guide/replication-controller.md#what-is-a-replication-controller
	Replicas int `json:"replicas"`

	// Selector is a label query over pods that should match the replicas count. If it is empty, it is defaulted to labels on Pod template; More info: http://releases.k8s.io/HEAD/docs/user-guide/labels.md#label-selectors
	Selector map[string]string `json:"selector,omitempty"`
}

// Scale subresource, applicable to ReplicationControllers and (in future) Deployment.
type Scale struct {
	api.TypeMeta `json:",inline"`
	// Standard object metadata; More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata.
	api.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the behavior of the scale. More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status.
	Spec ScaleSpec `json:"spec,omitempty"`

	// Status represents the current status of the scale. More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status. Read-only.
	Status ScaleStatus `json:"status,omitempty"`
}

// Dummy definition
type ReplicationControllerDummy struct {
	api.TypeMeta `json:",inline"`
}

// SubresourceReference contains enough information to let you inspect or modify the referred subresource.
type SubresourceReference struct {
	// Kind of the referent; More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds"
	Kind string `json:"kind,omitempty"`
	// Namespace of the referent; More info: http://releases.k8s.io/HEAD/docs/user-guide/namespaces.md
	Namespace string `json:"namespace,omitempty"`
	// Name of the referent; More info: http://releases.k8s.io/HEAD/docs/user-guide/identifiers.md#names
	Name string `json:"name,omitempty"`
	// API version of the referent
	APIVersion string `json:"apiVersion,omitempty"`
	// Subresource name of the referent
	Subresource string `json:"subresource,omitempty"`
}

// ResourceConsumption is an object for specifying average resource consumption of a particular resource.
type ResourceConsumption struct {
	Resource api.ResourceName  `json:"resource,omitempty"`
	Quantity resource.Quantity `json:"quantity,omitempty"`
}

// HorizontalPodAutoscalerSpec is the specification of a horizontal pod autoscaler.
type HorizontalPodAutoscalerSpec struct {
	// ScaleRef is a reference to Scale subresource. HorizontalPodAutoscaler will learn the current resource consumption from its status,
	// and will set the desired number of pods by modyfying its spec.
	ScaleRef *SubresourceReference `json:"scaleRef"`
	// MinCount is the lower limit for the number of pods that can be set by the autoscaler.
	MinCount int `json:"minCount"`
	// MaxCount is the upper limit for the number of pods that can be set by the autoscaler. It cannot be smaller than MinCount.
	MaxCount int `json:"maxCount"`
	// Target is the target average consumption of the given resource that the autoscaler will try to maintain by adjusting the desired number of pods.
	// Currently two types of resources are supported: "cpu" and "memory".
	Target ResourceConsumption `json:"target"`
}

// HorizontalPodAutoscalerStatus contains the current status of a horizontal pod autoscaler
type HorizontalPodAutoscalerStatus struct {
	// TODO: Consider if it is needed.
	// CurrentReplicas is the number of replicas of pods managed by this autoscaler.
	CurrentReplicas int `json:"currentReplicas"`

	// DesiredReplicas is the desired number of replicas of pods managed by this autoscaler.
	DesiredReplicas int `json:"desiredReplicas"`

	// CurrentConsumption is the current average consumption of the given resource that the autoscaler will
	// try to maintain by adjusting the desired number of pods.
	// Two types of resources are supported: "cpu" and "memory".
	CurrentConsumption *ResourceConsumption `json:"currentConsumption"`

	// LastScaleTimestamp is the last time the HorizontalPodAutoscaler scaled the number of pods.
	// This is used by the autoscaler to controll how often the number of pods is changed.
	LastScaleTimestamp *util.Time `json:"lastScaleTimestamp,omitempty"`
}

// HorizontalPodAutoscaler represents the configuration of a horizontal pod autoscaler.
type HorizontalPodAutoscaler struct {
	api.TypeMeta   `json:",inline"`
	api.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the behaviour of autoscaler. More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status.
	Spec HorizontalPodAutoscalerSpec `json:"spec,omitempty"`

	// Status represents the current information about the autoscaler.
	Status *HorizontalPodAutoscalerStatus `json:"status,omitempty"`
}

// HorizontalPodAutoscaler is a collection of pod autoscalers.
type HorizontalPodAutoscalerList struct {
	api.TypeMeta `json:",inline"`
	api.ListMeta `json:"metadata,omitempty"`

	// Items is the list of horizontal pod autoscalers.
	Items []HorizontalPodAutoscaler `json:"items"`
}

// A ThirdPartyResource is a generic representation of a resource, it is used by add-ons and plugins to add new resource
// types to the API.  It consists of one or more Versions of the api.
type ThirdPartyResource struct {
	api.TypeMeta `json:",inline"`

	// Standard object metadata
	api.ObjectMeta `json:"metadata,omitempty"`

	// Description is the description of this object.
	Description string `json:"description,omitempty"`

	// Versions are versions for this third party object
	Versions []APIVersion `json:"versions,omitempty"`
}

type ThirdPartyResourceList struct {
	api.TypeMeta `json:",inline"`

	// Standard list metadata.
	api.ListMeta `json:"metadata,omitempty"`

	// Items is the list of horizontal pod autoscalers.
	Items []ThirdPartyResource `json:"items"`
}

// An APIVersion represents a single concrete version of an object model.
type APIVersion struct {
	// Name of this version (e.g. 'v1').
	Name string `json:"name,omitempty"`

	// The API group to add this object into, default 'experimental'.
	APIGroup string `json:"apiGroup,omitempty"`
}

// An internal object, used for versioned storage in etcd.  Not exposed to the end user.
type ThirdPartyResourceData struct {
	api.TypeMeta `json:",inline"`
	// Standard object metadata.
	api.ObjectMeta `json:"metadata,omitempty"`

	// Data is the raw JSON data for this data.
	Data []byte `json:"name,omitempty"`
}

type Deployment struct {
	api.TypeMeta   `json:",inline"`
	api.ObjectMeta `json:"metadata,omitempty"`

	// Specification of the desired behavior of the Deployment.
	Spec DeploymentSpec `json:"spec,omitempty"`

	// Most recently observed status of the Deployment.
	Status DeploymentStatus `json:"status,omitempty"`
}

type DeploymentSpec struct {
	// Number of desired pods. This is a pointer to distinguish between explicit
	// zero and not specified. Defaults to 1.
	Replicas int `json:"replicas,omitempty"`

	// Label selector for pods. Existing ReplicationControllers whose pods are
	// selected by this will be scaled down.
	Selector map[string]string `json:"selector,omitempty"`

	// Template describes the pods that will be created.
	Template *api.PodTemplateSpec `json:"template,omitempty"`

	// The deployment strategy to use to replace existing pods with new ones.
	Strategy DeploymentStrategy `json:"strategy,omitempty"`

	// Key of the selector that is added to existing RCs (and label key that is
	// added to its pods) to prevent the existing RCs to select new pods (and old
	// pods being selected by new RC).
	// Users can set this to an empty string to indicate that the system should
	// not add any selector and label. If unspecified, system uses
	// "deployment.kubernetes.io/podTemplateHash".
	// Value of this key is hash of DeploymentSpec.PodTemplateSpec.
	// No label is added if this is set to empty string.
	UniqueLabelKey string `json:"uniqueLabelKey,omitempty"`
}

type DeploymentStrategy struct {
	// Type of deployment. Can be "Recreate" or "RollingUpdate". Default is RollingUpdate.
	Type DeploymentType `json:"type,omitempty"`

	// Rolling update config params. Present only if DeploymentType =
	// RollingUpdate.
	//---
	// TODO: Update this to follow our convention for oneOf, whatever we decide it
	// to be.
	RollingUpdate *RollingUpdateDeployment `json:"rollingUpdate,omitempty"`
}

type DeploymentType string

const (
	// Kill all existing pods before creating new ones.
	DeploymentRecreate DeploymentType = "Recreate"

	// Replace the old RCs by new one using rolling update i.e gradually scale down the old RCs and scale up the new one.
	DeploymentRollingUpdate DeploymentType = "RollingUpdate"
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
	MaxUnavailable util.IntOrString `json:"maxUnavailable,omitempty"`

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
	MaxSurge util.IntOrString `json:"maxSurge,omitempty"`

	// Minimum number of seconds for which a newly created pod should be ready
	// without any of its container crashing, for it to be considered available.
	// Defaults to 0 (pod will be considered available as soon as it is ready)
	MinReadySeconds int `json:"minReadySeconds,omitempty"`
}

type DeploymentStatus struct {
	// Total number of ready pods targeted by this deployment (this
	// includes both the old and new pods).
	Replicas int `json:"replicas,omitempty"`

	// Total number of new ready pods with the desired template spec.
	UpdatedReplicas int `json:"updatedReplicas,omitempty"`
}

type DeploymentList struct {
	api.TypeMeta `json:",inline"`
	api.ListMeta `json:"metadata,omitempty"`

	// Items is the list of deployments.
	Items []Deployment `json:"items"`
}

// DaemonSetSpec is the specification of a daemon set.
type DaemonSetSpec struct {
	// Selector is a label query over pods that are managed by the daemon set.
	// Must match in order to be controlled.
	// If empty, defaulted to labels on Pod template.
	// More info: http://releases.k8s.io/HEAD/docs/user-guide/labels.md#label-selectors
	Selector map[string]string `json:"selector,omitempty"`

	// Template is the object that describes the pod that will be created.
	// The DaemonSet will create exactly one copy of this pod on every node
	// that matches the template's node selector (or on every node if no node
	// selector is specified).
	// More info: http://releases.k8s.io/HEAD/docs/user-guide/replication-controller.md#pod-template
	Template *api.PodTemplateSpec `json:"template,omitempty"`
}

// DaemonSetStatus represents the current status of a daemon set.
type DaemonSetStatus struct {
	// CurrentNumberScheduled is the number of nodes that are running exactly 1
	// daemon pod and are supposed to run the daemon pod.
	CurrentNumberScheduled int `json:"currentNumberScheduled"`

	// NumberMisscheduled is the number of nodes that are running the daemon pod, but are
	// not supposed to run the daemon pod.
	NumberMisscheduled int `json:"numberMisscheduled"`

	// DesiredNumberScheduled is the total number of nodes that should be running the daemon
	// pod (including nodes correctly running the daemon pod).
	DesiredNumberScheduled int `json:"desiredNumberScheduled"`
}

// DaemonSet represents the configuration of a daemon set.
type DaemonSet struct {
	api.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	api.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the desired behavior of this daemon set.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
	Spec DaemonSetSpec `json:"spec,omitempty"`

	// Status is the current status of this daemon set. This data may be
	// out of date by some window of time.
	// Populated by the system.
	// Read-only.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
	Status DaemonSetStatus `json:"status,omitempty"`
}

// DaemonSetList is a collection of daemon sets.
type DaemonSetList struct {
	api.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	api.ListMeta `json:"metadata,omitempty"`

	// Items is a list of daemon sets.
	Items []DaemonSet `json:"items"`
}

type ThirdPartyResourceDataList struct {
	api.TypeMeta `json:",inline"`
	// Standard list metadata
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	api.ListMeta `json:"metadata,omitempty"`
	// Items is a list of third party objects
	Items []ThirdPartyResourceData `json:"items"`
}

// Job represents the configuration of a single job.
type Job struct {
	api.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	api.ObjectMeta `json:"metadata,omitempty"`

	// Spec is a structure defining the expected behavior of a job.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
	Spec JobSpec `json:"spec,omitempty"`

	// Status is a structure describing current status of a job.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
	Status JobStatus `json:"status,omitempty"`
}

// JobList is a collection of jobs.
type JobList struct {
	api.TypeMeta `json:",inline"`
	// Standard list metadata
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	api.ListMeta `json:"metadata,omitempty"`

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
	// job should be run with. Defaults to 1.
	Completions *int `json:"completions,omitempty"`

	// Selector is a label query over pods that should match the pod count.
	Selector map[string]string `json:"selector"`

	// Template is the object that describes the pod that will be created when
	// executing a job.
	Template *api.PodTemplateSpec `json:"template"`
}

// JobStatus represents the current state of a Job.
type JobStatus struct {

	// Conditions represent the latest available observations of an object's current state.
	Conditions []JobCondition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`

	// StartTime represents time when the job was acknowledged by the Job Manager.
	// It is not guaranteed to be set in happens-before order across separate operations.
	// It is represented in RFC3339 form and is in UTC.
	StartTime *util.Time `json:"startTime,omitempty"`

	// CompletionTime represents time when the job was completed. It is not guaranteed to
	// be set in happens-before order across separate operations.
	// It is represented in RFC3339 form and is in UTC.
	CompletionTime *util.Time `json:"completionTime,omitempty"`

	// Active is the number of actively running pods.
	Active int `json:"active,omitempty"`

	// Successful is the number of pods which reached Phase Succeeded.
	Successful int `json:"successful,omitempty"`

	// Unsuccessful is the number of pods failures, this applies only to jobs
	// created with RestartPolicyNever, otherwise this value will always be 0.
	Unsuccessful int `json:"unsuccessful,omitempty"`
}

type JobConditionType string

// These are valid conditions of a job.
const (
	// JobComplete means the job has completed its execution.
	JobComplete JobConditionType = "Complete"
)

// JobCondition describes current state of a job.
type JobCondition struct {
	// Type of job condition, currently only Complete.
	Type JobConditionType `json:"type"`
	// Status of the condition, one of True, False, Unknown.
	Status api.ConditionStatus `json:"status"`
	// Last time the condition was checked.
	LastProbeTime util.Time `json:"lastProbeTime,omitempty"`
	// Last time the condition transit from one status to another.
	LastTransitionTime util.Time `json:"lastTransitionTime,omitempty"`
	// (brief) reason for the condition's last transition.
	Reason string `json:"reason,omitempty"`
	// Human readable message indicating details about last transition.
	Message string `json:"message,omitempty"`
}
