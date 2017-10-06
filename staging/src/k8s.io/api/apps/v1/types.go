/*
Copyright 2017 The Kubernetes Authors.

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
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

const (
	ControllerRevisionHashLabelKey = "controller-revision-hash"
	DeprecatedTemplateGeneration   = "deprecated.daemonset.template.generation"
)

// DaemonSetUpdateStrategy is a struct used to control the update strategy for a DaemonSet.
type DaemonSetUpdateStrategy struct {
	// Type of daemon set update. Can be "RollingUpdate" or "OnDelete". Default is RollingUpdate.
	// +optional
	Type DaemonSetUpdateStrategyType `json:"type,omitempty" protobuf:"bytes,1,opt,name=type"`

	// Rolling update config params. Present only if type = "RollingUpdate".
	//---
	// TODO: Update this to follow our convention for oneOf, whatever we decide it
	// to be. Same as Deployment `strategy.rollingUpdate`.
	// See https://github.com/kubernetes/kubernetes/issues/35345
	// +optional
	RollingUpdate *RollingUpdateDaemonSet `json:"rollingUpdate,omitempty" protobuf:"bytes,2,opt,name=rollingUpdate"`
}

type DaemonSetUpdateStrategyType string

const (
	// Replace the old daemons by new ones using rolling update i.e replace them on each node one after the other.
	RollingUpdateDaemonSetStrategyType DaemonSetUpdateStrategyType = "RollingUpdate"

	// Replace the old daemons only when it's killed
	OnDeleteDaemonSetStrategyType DaemonSetUpdateStrategyType = "OnDelete"
)

// Spec to control the desired behavior of daemon set rolling update.
type RollingUpdateDaemonSet struct {
	// The maximum number of DaemonSet pods that can be unavailable during the
	// update. Value can be an absolute number (ex: 5) or a percentage of total
	// number of DaemonSet pods at the start of the update (ex: 10%). Absolute
	// number is calculated from percentage by rounding up.
	// This cannot be 0.
	// Default value is 1.
	// Example: when this is set to 30%, at most 30% of the total number of nodes
	// that should be running the daemon pod (i.e. status.desiredNumberScheduled)
	// can have their pods stopped for an update at any given
	// time. The update starts by stopping at most 30% of those DaemonSet pods
	// and then brings up new DaemonSet pods in their place. Once the new pods
	// are available, it then proceeds onto other DaemonSet pods, thus ensuring
	// that at least 70% of original number of DaemonSet pods are available at
	// all times during the update.
	// +optional
	MaxUnavailable *intstr.IntOrString `json:"maxUnavailable,omitempty" protobuf:"bytes,1,opt,name=maxUnavailable"`
}

// DaemonSetSpec is the specification of a daemon set.
type DaemonSetSpec struct {
	// A label query over pods that are managed by the daemon set.
	// Must match in order to be controlled.
	// If empty, defaulted to labels on Pod template.
	// More info: https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#label-selectors
	// +optional
	Selector *metav1.LabelSelector `json:"selector,omitempty" protobuf:"bytes,1,opt,name=selector"`

	// An object that describes the pod that will be created.
	// The DaemonSet will create exactly one copy of this pod on every node
	// that matches the template's node selector (or on every node if no node
	// selector is specified).
	// More info: https://kubernetes.io/docs/concepts/workloads/controllers/replicationcontroller#pod-template
	Template v1.PodTemplateSpec `json:"template" protobuf:"bytes,2,opt,name=template"`

	// An update strategy to replace existing DaemonSet pods with new pods.
	// +optional
	UpdateStrategy DaemonSetUpdateStrategy `json:"updateStrategy,omitempty" protobuf:"bytes,3,opt,name=updateStrategy"`

	// The minimum number of seconds for which a newly created DaemonSet pod should
	// be ready without any of its container crashing, for it to be considered
	// available. Defaults to 0 (pod will be considered available as soon as it
	// is ready).
	// +optional
	MinReadySeconds int32 `json:"minReadySeconds,omitempty" protobuf:"varint,4,opt,name=minReadySeconds"`

	// The number of old history to retain to allow rollback.
	// This is a pointer to distinguish between explicit zero and not specified.
	// Defaults to 10.
	// +optional
	RevisionHistoryLimit *int32 `json:"revisionHistoryLimit,omitempty" protobuf:"varint,6,opt,name=revisionHistoryLimit"`
}

// DaemonSetStatus represents the current status of a daemon set.
type DaemonSetStatus struct {
	// The number of nodes that are running at least 1
	// daemon pod and are supposed to run the daemon pod.
	// More info: https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/
	CurrentNumberScheduled int32 `json:"currentNumberScheduled" protobuf:"varint,1,opt,name=currentNumberScheduled"`

	// The number of nodes that are running the daemon pod, but are
	// not supposed to run the daemon pod.
	// More info: https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/
	NumberMisscheduled int32 `json:"numberMisscheduled" protobuf:"varint,2,opt,name=numberMisscheduled"`

	// The total number of nodes that should be running the daemon
	// pod (including nodes correctly running the daemon pod).
	// More info: https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/
	DesiredNumberScheduled int32 `json:"desiredNumberScheduled" protobuf:"varint,3,opt,name=desiredNumberScheduled"`

	// The number of nodes that should be running the daemon pod and have one
	// or more of the daemon pod running and ready.
	NumberReady int32 `json:"numberReady" protobuf:"varint,4,opt,name=numberReady"`

	// The most recent generation observed by the daemon set controller.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty" protobuf:"varint,5,opt,name=observedGeneration"`

	// The total number of nodes that are running updated daemon pod
	// +optional
	UpdatedNumberScheduled int32 `json:"updatedNumberScheduled,omitempty" protobuf:"varint,6,opt,name=updatedNumberScheduled"`

	// The number of nodes that should be running the
	// daemon pod and have one or more of the daemon pod running and
	// available (ready for at least spec.minReadySeconds)
	// +optional
	NumberAvailable int32 `json:"numberAvailable,omitempty" protobuf:"varint,7,opt,name=numberAvailable"`

	// The number of nodes that should be running the
	// daemon pod and have none of the daemon pod running and available
	// (ready for at least spec.minReadySeconds)
	// +optional
	NumberUnavailable int32 `json:"numberUnavailable,omitempty" protobuf:"varint,8,opt,name=numberUnavailable"`

	// Count of hash collisions for the DaemonSet. The DaemonSet controller
	// uses this field as a collision avoidance mechanism when it needs to
	// create the name for the newest ControllerRevision.
	// +optional
	CollisionCount *int32 `json:"collisionCount,omitempty" protobuf:"varint,9,opt,name=collisionCount"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DaemonSet represents the configuration of a daemon set.
type DaemonSet struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// The desired behavior of this daemon set.
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#spec-and-status
	// +optional
	Spec DaemonSetSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// The current status of this daemon set. This data may be
	// out of date by some window of time.
	// Populated by the system.
	// Read-only.
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#spec-and-status
	// +optional
	Status DaemonSetStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

const (
	// DefaultDaemonSetUniqueLabelKey is the default label key that is added
	// to existing DaemonSet pods to distinguish between old and new
	// DaemonSet pods during DaemonSet template updates.
	DefaultDaemonSetUniqueLabelKey = ControllerRevisionHashLabelKey
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DaemonSetList is a collection of daemon sets.
type DaemonSetList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// A list of daemon sets.
	Items []DaemonSet `json:"items" protobuf:"bytes,2,rep,name=items"`
}
