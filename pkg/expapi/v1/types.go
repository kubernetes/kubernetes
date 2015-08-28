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

package v1

import (
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util"
)

// ScaleSpec describes the attributes a Scale subresource
type ScaleSpec struct {
	// Replicas is the number of desired replicas.
	Replicas int `json:"replicas,omitempty" description:"number of replicas desired; see http://releases.k8s.io/HEAD/docs/user-guide/replication-controller.md#what-is-a-replication-controller"`
}

// ScaleStatus represents the current status of a Scale subresource.
type ScaleStatus struct {
	// Replicas is the number of actual replicas.
	Replicas int `json:"replicas" description:"most recently oberved number of replicas; see http://releases.k8s.io/HEAD/docs/user-guide/replication-controller.md#what-is-a-replication-controller"`

	// Selector is a label query over pods that should match the replicas count.
	Selector map[string]string `json:"selector,omitempty" description:"label keys and values that must match in order to be controlled by this replication controller, if empty defaulted to labels on Pod template; see http://releases.k8s.io/HEAD/docs/user-guide/labels.md#label-selectors"`
}

// Scale subresource, applicable to ReplicationControllers and (in future) Deployment.
type Scale struct {
	v1.TypeMeta   `json:",inline"`
	v1.ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata"`

	// Spec defines the behavior of the scale.
	Spec ScaleSpec `json:"spec,omitempty" description:"specification of the desired behavior of the scale; http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status"`

	// Status represents the current status of the scale.
	Status ScaleStatus `json:"status,omitempty" description:"most recently observed status of the service; populated by the system, read-only; http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status"`
}

// Dummy definition
type ReplicationControllerDummy struct {
	v1.TypeMeta `json:",inline"`
}

// SubresourceReference contains enough information to let you inspect or modify the referred subresource.
type SubresourceReference struct {
	Kind        string `json:"kind,omitempty" description:"kind of the referent; see http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds"`
	Namespace   string `json:"namespace,omitempty" description:"namespace of the referent; see http://releases.k8s.io/HEAD/docs/user-guide/namespaces.md"`
	Name        string `json:"name,omitempty" description:"name of the referent; see http://releases.k8s.io/HEAD/docs/user-guide/identifiers.md#names"`
	APIVersion  string `json:"apiVersion,omitempty" description:"API version of the referent"`
	Subresource string `json:"subresource,omitempty" decription:"subresource name of the referent"`
}

// ResourceConsumption is an object for specifying average resource consumption of a particular resource.
type ResourceConsumption struct {
	Resource v1.ResourceName   `json:"resource,omitempty"`
	Quantity resource.Quantity `json:"quantity,omitempty"`
}

// HorizontalPodAutoscalerSpec is the specification of a horizontal pod autoscaler.
type HorizontalPodAutoscalerSpec struct {
	// ScaleRef is a reference to Scale subresource. HorizontalPodAutoscaler will learn the current resource consumption from its status,
	// and will set the desired number of pods by modyfying its spec.
	ScaleRef *SubresourceReference `json:"scaleRef" description:"reference to scale subresource for quering the current resource cosumption and for setting the desired number of pods"`
	// MinCount is the lower limit for the number of pods that can be set by the autoscaler.
	MinCount int `json:"minCount" description:"lower limit for the number of pods"`
	// MaxCount is the upper limit for the number of pods that can be set by the autoscaler. It cannot be smaller than MinCount.
	MaxCount int `json:"maxCount" description:"upper limit for the number of pods"`
	// Target is the target average consumption of the given resource that the autoscaler will try to maintain by adjusting the desired number of pods.
	// Currently two types of resources are supported: "cpu" and "memory".
	Target ResourceConsumption `json:"target" description:"target average consumption of resource that the autoscaler will try to maintain by adjusting the desired number of pods"`
}

// HorizontalPodAutoscalerStatus contains the current status of a horizontal pod autoscaler
type HorizontalPodAutoscalerStatus struct {
	// CurrentReplicas is the number of replicas of pods managed by this autoscaler.
	CurrentReplicas int `json:"currentReplicas" description:"number of replicas observed by the autoscaler"`

	// DesiredReplicas is the desired number of replicas of pods managed by this autoscaler.
	DesiredReplicas int `json:"desiredReplicas" description:"number of desired replicas"`

	// CurrentConsumption is the current average consumption of the given resource that the autoscaler will
	// try to maintain by adjusting the desired number of pods.
	// Two types of resources are supported: "cpu" and "memory".
	CurrentConsumption *ResourceConsumption `json:"currentConsumption" description:"current resource consumption"`

	// LastScaleTimestamp is the last time the HorizontalPodAutoscaler scaled the number of pods.
	// This is used by the autoscaler to controll how often the number of pods is changed.
	LastScaleTimestamp *util.Time `json:"lastScaleTimestamp,omitempty" description:"last time the autoscaler made decision about changing the number of pods"`
}

// HorizontalPodAutoscaler represents the configuration of a horizontal pod autoscaler.
type HorizontalPodAutoscaler struct {
	v1.TypeMeta   `json:",inline"`
	v1.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the behaviour of autoscaler.
	Spec HorizontalPodAutoscalerSpec `json:"spec,omitempty" description:"specification of the desired behavior of the autoscaler; http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status"`

	// Status represents the current information about the autoscaler.
	Status *HorizontalPodAutoscalerStatus `json:"status,omitempty"`
}

// HorizontalPodAutoscaler is a collection of pod autoscalers.
type HorizontalPodAutoscalerList struct {
	v1.TypeMeta `json:",inline"`
	v1.ListMeta `json:"metadata,omitempty"`

	Items []HorizontalPodAutoscaler `json:"items" description:"list of horizontal pod autoscalers"`
}

// A ThirdPartyResource is a generic representation of a resource, it is used by add-ons and plugins to add new resource
// types to the API.  It consists of one or more Versions of the api.
type ThirdPartyResource struct {
	v1.TypeMeta   `json:",inline"`
	v1.ObjectMeta `json:"metadata,omitempty" description:"standard object metadata"`

	Description string `json:"description,omitempty" description:"The description of this object"`

	Versions []APIVersion `json:"versions,omitempty" description:"The versions for this third party object"`
}

type ThirdPartyResourceList struct {
	v1.TypeMeta `json:",inline"`
	v1.ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://docs.k8s.io/api-conventions.md#metadata"`

	Items []ThirdPartyResource `json:"items" description:"items is a list of schema objects"`
}

// An APIVersion represents a single concrete version of an object model.
type APIVersion struct {
	Name     string `json:"name,omitempty" description:"name of this version (e.g. 'v1')"`
	APIGroup string `json:"apiGroup,omitempty" description:"The API group to add this object into, default 'experimental'"`
}

// An internal object, used for versioned storage in etcd.  Not exposed to the end user.
type ThirdPartyResourceData struct {
	v1.TypeMeta   `json:",inline"`
	v1.ObjectMeta `json:"metadata,omitempty" description:"standard object metadata"`

	Data []byte `json:"name,omitempty" description:"the raw JSON data for this data"`
}

type Deployment struct {
	v1.TypeMeta   `json:",inline"`
	v1.ObjectMeta `json:"metadata,omitempty"`

	// Specification of the desired behavior of the Deployment.
	Spec DeploymentSpec `json:"spec,omitempty" description: "specification of the desired behavior of deployment"`

	// Most recently observed status of the Deployment.
	Status DeploymentStatus `json:"status,omitempty" description: "most recently observed status of deployment"`
}

type DeploymentSpec struct {
	// Number of desired pods. This is a pointer to distinguish between explicit
	// zero and not specified. Defaults to 1.
	Replicas *int `json:"replicas,omitempty" description:"number of desired pods; Defaults to 1"`

	// Label selector for pods. Existing ReplicationControllers whose pods are
	// selected by this will be scaled down.
	Selector map[string]string `json:"selector,omitempty" description:"label selector for pods; existing replication controllers whose pods are selected by this will be scaled down`

	// Describes the pods that will be created.
	Template *v1.PodTemplateSpec `json:"template,omitempty" description:"template to describe the pods that will be created"`

	// The deployment strategy to use to replace existing pods with new ones.
	Strategy DeploymentStrategy `json:"strategy,omitempty" description:"deployment strategy to use to replace existing pods with new ones"`

	// Key of the selector that is added to existing RCs (and label key that is
	// added to its pods) to prevent the existing RCs to select new pods (and old
	// pods being selected by new RC).
	// Users can set this to an empty string to indicate that the system should
	// not add any selector and label. If unspecified, system uses
	// "deployment.kubernetes.io/podTemplateHash".
	// Value of this key is hash of DeploymentSpec.PodTemplateSpec.
	UniqueLabelKey *string `json:"uniqueLabel,omitempty" description:"key of the label that is added to existing pods to distinguish them from new ones; deployment.kubernetes.io/podTemplateHash is used by default; value of this label is hash of pod template spec; no label is added if this is set to empty string"`
}

type DeploymentStrategy struct {
	// Type of deployment. Can be "Recreate" or "RollingUpdate". Default is RollingUpdate.
	Type DeploymentType `json:"type,omitempty" description:"type of deployment; can be Recreate or RollingUpdate; defaults to RollingUpdate"`

	// TODO: Update this to follow our convention for oneOf, whatever we decide it
	// to be.
	// Rolling update config params. Present only if DeploymentType =
	// RollingUpdate.
	RollingUpdate *RollingUpdateDeployment `json:"rollingUpdate,omitempty" description:"rolling update config params; present only if deploymentType=RollingUpdate"`
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
	MaxUnavailable util.IntOrString `json:"maxUnavailable,omitempty" description:"max number of pods that can be unavailable during the update; value can be an absolute number or a percentage of total pods at start of update"`

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
	MaxSurge util.IntOrString `json:"maxSurge,omitempty" description:"max number of pods that can be scheduled above the original number of pods; value can be an absolute number or a percentage of total pods at start of update"`

	// Minimum number of seconds for which a newly created pod should be ready
	// without any of its container crashing, for it to be considered available.
	// Defaults to 0 (pod will be considered available as soon as it is ready)
	MinReadySeconds int `json:"minReadySeconds,omitempty" description:"min seconds for which a newly created pod should be ready for it to be considered available"`
}

type DeploymentStatus struct {
	// Total number of ready pods targeted by this deployment (this
	// includes both the old and new pods).
	Replicas int `json:"replicas,omitempty" description:"total number of ready pods targeted by this deployment (includes both old and new pods"`

	// Total number of new ready pods with the desired template spec.
	UpdatedReplicas int `json:"updatedReplicas,omitempty" description:"total number of new ready pods with the desired template spec"`
}

type DeploymentList struct {
	v1.TypeMeta `json:",inline"`
	v1.ListMeta `json:"metadata,omitempty"`

	Items []Deployment `json:"items" description:"list of deployments"`
}

// DaemonSpec is the specification of a daemon.
type DaemonSpec struct {
	// Selector is a label query over pods that are managed by the daemon.
	// Must match in order to be controlled.
	// If empty, defaulted to labels on Pod template.
	// More info: http://releases.k8s.io/HEAD/docs/user-guide/labels.md#label-selectors
	Selector map[string]string `json:"selector,omitempty"`

	// Template is the object that describes the pod that will be created.
	// The Daemon will create exactly one copy of this pod on every node
	// that matches the template's node selector (or on every node if no node
	// selector is specified).
	// More info: http://releases.k8s.io/HEAD/docs/user-guide/replication-controller.md#pod-template
	Template *v1.PodTemplateSpec `json:"template,omitempty"`
}

// DaemonStatus represents the current status of a daemon.
type DaemonStatus struct {
	// CurrentNumberScheduled is the number of nodes that are running exactly 1 copy of the
	// daemon and are supposed to run the daemon.
	CurrentNumberScheduled int `json:"currentNumberScheduled"`

	// NumberMisscheduled is the number of nodes that are running the daemon, but are
	// not supposed to run the daemon.
	NumberMisscheduled int `json:"numberMisscheduled"`

	// DesiredNumberScheduled is the total number of nodes that should be running the daemon
	// (including nodes correctly running the daemon).
	DesiredNumberScheduled int `json:"desiredNumberScheduled"`
}

// Daemon represents the configuration of a daemon.
type Daemon struct {
	v1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	v1.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the specification of the desired behavior of this daemon.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
	Spec DaemonSpec `json:"spec,omitempty"`

	// Status is the current status of this daemon. This data may be
	// out of date by some window of time.
	// Populated by the system.
	// Read-only.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
	Status DaemonStatus `json:"status,omitempty"`
}

// DaemonList is a collection of daemon.
type DaemonList struct {
	v1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	v1.ListMeta `json:"metadata,omitempty"`

	// Items is a list of daemons.
	Items []Daemon `json:"items"`
}
