/*
Copyright 2015 The Kubernetes Authors.

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

package v1beta1

import (
	appsv1beta1 "k8s.io/api/apps/v1beta1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

// describes the attributes of a scale subresource
type ScaleSpec struct {
	// desired number of instances for the scaled object.
	// +optional
	Replicas int32 `json:"replicas,omitempty" protobuf:"varint,1,opt,name=replicas"`
}

// represents the current status of a scale subresource.
type ScaleStatus struct {
	// actual number of observed instances of the scaled object.
	Replicas int32 `json:"replicas" protobuf:"varint,1,opt,name=replicas"`

	// label query over pods that should match the replicas count. More info: http://kubernetes.io/docs/user-guide/labels#label-selectors
	// +optional
	Selector map[string]string `json:"selector,omitempty" protobuf:"bytes,2,rep,name=selector"`

	// label selector for pods that should match the replicas count. This is a serializated
	// version of both map-based and more expressive set-based selectors. This is done to
	// avoid introspection in the clients. The string will be in the same format as the
	// query-param syntax. If the target type only supports map-based selectors, both this
	// field and map-based selector field are populated.
	// More info: https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#label-selectors
	// +optional
	TargetSelector string `json:"targetSelector,omitempty" protobuf:"bytes,3,opt,name=targetSelector"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// represents a scaling request for a resource.
type Scale struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata; More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// defines the behavior of the scale. More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status.
	// +optional
	Spec ScaleSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// current status of the scale. More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status. Read-only.
	// +optional
	Status ScaleStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Dummy definition
type ReplicationControllerDummy struct {
	metav1.TypeMeta `json:",inline"`
}

// +genclient
// +genclient:method=GetScale,verb=get,subresource=scale,result=Scale
// +genclient:method=UpdateScale,verb=update,subresource=scale,input=Scale,result=Scale
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DEPRECATED - This group version of Deployment is deprecated by apps/v1beta2/Deployment. See the release notes for
// more information.
// Deployment enables declarative updates for Pods and ReplicaSets.
type Deployment struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Specification of the desired behavior of the Deployment.
	// +optional
	Spec DeploymentSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// Most recently observed status of the Deployment.
	// +optional
	Status DeploymentStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// DeploymentSpec is the specification of the desired behavior of the Deployment.
type DeploymentSpec struct {
	// Number of desired pods. This is a pointer to distinguish between explicit
	// zero and not specified. Defaults to 1.
	// +optional
	Replicas *int32 `json:"replicas,omitempty" protobuf:"varint,1,opt,name=replicas"`

	// Label selector for pods. Existing ReplicaSets whose pods are
	// selected by this will be the ones affected by this deployment.
	// +optional
	Selector *metav1.LabelSelector `json:"selector,omitempty" protobuf:"bytes,2,opt,name=selector"`

	// Template describes the pods that will be created.
	Template v1.PodTemplateSpec `json:"template" protobuf:"bytes,3,opt,name=template"`

	// The deployment strategy to use to replace existing pods with new ones.
	// +optional
	// +patchStrategy=retainKeys
	Strategy DeploymentStrategy `json:"strategy,omitempty" patchStrategy:"retainKeys" protobuf:"bytes,4,opt,name=strategy"`

	// Minimum number of seconds for which a newly created pod should be ready
	// without any of its container crashing, for it to be considered available.
	// Defaults to 0 (pod will be considered available as soon as it is ready)
	// +optional
	MinReadySeconds int32 `json:"minReadySeconds,omitempty" protobuf:"varint,5,opt,name=minReadySeconds"`

	// The number of old ReplicaSets to retain to allow rollback.
	// This is a pointer to distinguish between explicit zero and not specified.
	// This is set to the max value of int32 (i.e. 2147483647) by default, which
	// means "retaining all old RelicaSets".
	// +optional
	RevisionHistoryLimit *int32 `json:"revisionHistoryLimit,omitempty" protobuf:"varint,6,opt,name=revisionHistoryLimit"`

	// Indicates that the deployment is paused and will not be processed by the
	// deployment controller.
	// +optional
	Paused bool `json:"paused,omitempty" protobuf:"varint,7,opt,name=paused"`

	// DEPRECATED.
	// The config this deployment is rolling back to. Will be cleared after rollback is done.
	// +optional
	RollbackTo *RollbackConfig `json:"rollbackTo,omitempty" protobuf:"bytes,8,opt,name=rollbackTo"`

	// The maximum time in seconds for a deployment to make progress before it
	// is considered to be failed. The deployment controller will continue to
	// process failed deployments and a condition with a ProgressDeadlineExceeded
	// reason will be surfaced in the deployment status. Note that progress will
	// not be estimated during the time a deployment is paused. This is set to
	// the max value of int32 (i.e. 2147483647) by default, which means "no deadline".
	// +optional
	ProgressDeadlineSeconds *int32 `json:"progressDeadlineSeconds,omitempty" protobuf:"varint,9,opt,name=progressDeadlineSeconds"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DEPRECATED.
// DeploymentRollback stores the information required to rollback a deployment.
type DeploymentRollback struct {
	metav1.TypeMeta `json:",inline"`
	// Required: This must match the Name of a deployment.
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`
	// The annotations to be updated to a deployment
	// +optional
	UpdatedAnnotations map[string]string `json:"updatedAnnotations,omitempty" protobuf:"bytes,2,rep,name=updatedAnnotations"`
	// The config of this deployment rollback.
	RollbackTo RollbackConfig `json:"rollbackTo" protobuf:"bytes,3,opt,name=rollbackTo"`
}

// DEPRECATED.
type RollbackConfig struct {
	// The revision to rollback to. If set to 0, rollback to the last revision.
	// +optional
	Revision int64 `json:"revision,omitempty" protobuf:"varint,1,opt,name=revision"`
}

const (
	// DefaultDeploymentUniqueLabelKey is the default key of the selector that is added
	// to existing RCs (and label key that is added to its pods) to prevent the existing RCs
	// to select new pods (and old pods being select by new RC).
	DefaultDeploymentUniqueLabelKey string = "pod-template-hash"
)

// DeploymentStrategy describes how to replace existing pods with new ones.
type DeploymentStrategy struct {
	// Type of deployment. Can be "Recreate" or "RollingUpdate". Default is RollingUpdate.
	// +optional
	Type DeploymentStrategyType `json:"type,omitempty" protobuf:"bytes,1,opt,name=type,casttype=DeploymentStrategyType"`

	// Rolling update config params. Present only if DeploymentStrategyType =
	// RollingUpdate.
	//---
	// TODO: Update this to follow our convention for oneOf, whatever we decide it
	// to be.
	// +optional
	RollingUpdate *RollingUpdateDeployment `json:"rollingUpdate,omitempty" protobuf:"bytes,2,opt,name=rollingUpdate"`
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
	// Value can be an absolute number (ex: 5) or a percentage of desired pods (ex: 10%).
	// Absolute number is calculated from percentage by rounding down.
	// This can not be 0 if MaxSurge is 0.
	// By default, a fixed value of 1 is used.
	// Example: when this is set to 30%, the old RC can be scaled down to 70% of desired pods
	// immediately when the rolling update starts. Once new pods are ready, old RC
	// can be scaled down further, followed by scaling up the new RC, ensuring
	// that the total number of pods available at all times during the update is at
	// least 70% of desired pods.
	// +optional
	MaxUnavailable *intstr.IntOrString `json:"maxUnavailable,omitempty" protobuf:"bytes,1,opt,name=maxUnavailable"`

	// The maximum number of pods that can be scheduled above the desired number of
	// pods.
	// Value can be an absolute number (ex: 5) or a percentage of desired pods (ex: 10%).
	// This can not be 0 if MaxUnavailable is 0.
	// Absolute number is calculated from percentage by rounding up.
	// By default, a value of 1 is used.
	// Example: when this is set to 30%, the new RC can be scaled up immediately when
	// the rolling update starts, such that the total number of old and new pods do not exceed
	// 130% of desired pods. Once old pods have been killed,
	// new RC can be scaled up further, ensuring that total number of pods running
	// at any time during the update is at most 130% of desired pods.
	// +optional
	MaxSurge *intstr.IntOrString `json:"maxSurge,omitempty" protobuf:"bytes,2,opt,name=maxSurge"`
}

// DeploymentStatus is the most recently observed status of the Deployment.
type DeploymentStatus struct {
	// The generation observed by the deployment controller.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty" protobuf:"varint,1,opt,name=observedGeneration"`

	// Total number of non-terminated pods targeted by this deployment (their labels match the selector).
	// +optional
	Replicas int32 `json:"replicas,omitempty" protobuf:"varint,2,opt,name=replicas"`

	// Total number of non-terminated pods targeted by this deployment that have the desired template spec.
	// +optional
	UpdatedReplicas int32 `json:"updatedReplicas,omitempty" protobuf:"varint,3,opt,name=updatedReplicas"`

	// Total number of ready pods targeted by this deployment.
	// +optional
	ReadyReplicas int32 `json:"readyReplicas,omitempty" protobuf:"varint,7,opt,name=readyReplicas"`

	// Total number of available pods (ready for at least minReadySeconds) targeted by this deployment.
	// +optional
	AvailableReplicas int32 `json:"availableReplicas,omitempty" protobuf:"varint,4,opt,name=availableReplicas"`

	// Total number of unavailable pods targeted by this deployment. This is the total number of
	// pods that are still required for the deployment to have 100% available capacity. They may
	// either be pods that are running but not yet available or pods that still have not been created.
	// +optional
	UnavailableReplicas int32 `json:"unavailableReplicas,omitempty" protobuf:"varint,5,opt,name=unavailableReplicas"`

	// Represents the latest available observations of a deployment's current state.
	// +patchMergeKey=type
	// +patchStrategy=merge
	Conditions []DeploymentCondition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,6,rep,name=conditions"`

	// Count of hash collisions for the Deployment. The Deployment controller uses this
	// field as a collision avoidance mechanism when it needs to create the name for the
	// newest ReplicaSet.
	// +optional
	CollisionCount *int32 `json:"collisionCount,omitempty" protobuf:"varint,8,opt,name=collisionCount"`
}

type DeploymentConditionType string

// These are valid conditions of a deployment.
const (
	// Available means the deployment is available, ie. at least the minimum available
	// replicas required are up and running for at least minReadySeconds.
	DeploymentAvailable DeploymentConditionType = "Available"
	// Progressing means the deployment is progressing. Progress for a deployment is
	// considered when a new replica set is created or adopted, and when new pods scale
	// up or old pods scale down. Progress is not estimated for paused deployments or
	// when progressDeadlineSeconds is not specified.
	DeploymentProgressing DeploymentConditionType = "Progressing"
	// ReplicaFailure is added in a deployment when one of its pods fails to be created
	// or deleted.
	DeploymentReplicaFailure DeploymentConditionType = "ReplicaFailure"
)

// DeploymentCondition describes the state of a deployment at a certain point.
type DeploymentCondition struct {
	// Type of deployment condition.
	Type DeploymentConditionType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=DeploymentConditionType"`
	// Status of the condition, one of True, False, Unknown.
	Status v1.ConditionStatus `json:"status" protobuf:"bytes,2,opt,name=status,casttype=k8s.io/api/core/v1.ConditionStatus"`
	// The last time this condition was updated.
	LastUpdateTime metav1.Time `json:"lastUpdateTime,omitempty" protobuf:"bytes,6,opt,name=lastUpdateTime"`
	// Last time the condition transitioned from one status to another.
	LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty" protobuf:"bytes,7,opt,name=lastTransitionTime"`
	// The reason for the condition's last transition.
	Reason string `json:"reason,omitempty" protobuf:"bytes,4,opt,name=reason"`
	// A human readable message indicating details about the transition.
	Message string `json:"message,omitempty" protobuf:"bytes,5,opt,name=message"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DeploymentList is a list of Deployments.
type DeploymentList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of Deployments.
	Items []Deployment `json:"items" protobuf:"bytes,2,rep,name=items"`
}

type DaemonSetUpdateStrategy struct {
	// Type of daemon set update. Can be "RollingUpdate" or "OnDelete".
	// Default is OnDelete.
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

	// DEPRECATED.
	// A sequence number representing a specific generation of the template.
	// Populated by the system. It can be set only during the creation.
	// +optional
	TemplateGeneration int64 `json:"templateGeneration,omitempty" protobuf:"varint,5,opt,name=templateGeneration"`

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

	// Represents the latest available observations of a DaemonSet's current state.
	// +optional
	// +patchMergeKey=type
	// +patchStrategy=merge
	Conditions []DaemonSetCondition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,10,rep,name=conditions"`
}

type DaemonSetConditionType string

// TODO: Add valid condition types of a DaemonSet.

// DaemonSetCondition describes the state of a DaemonSet at a certain point.
type DaemonSetCondition struct {
	// Type of DaemonSet condition.
	Type DaemonSetConditionType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=DaemonSetConditionType"`
	// Status of the condition, one of True, False, Unknown.
	Status v1.ConditionStatus `json:"status" protobuf:"bytes,2,opt,name=status,casttype=k8s.io/api/core/v1.ConditionStatus"`
	// Last time the condition transitioned from one status to another.
	// +optional
	LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty" protobuf:"bytes,3,opt,name=lastTransitionTime"`
	// The reason for the condition's last transition.
	// +optional
	Reason string `json:"reason,omitempty" protobuf:"bytes,4,opt,name=reason"`
	// A human readable message indicating details about the transition.
	// +optional
	Message string `json:"message,omitempty" protobuf:"bytes,5,opt,name=message"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DEPRECATED - This group version of DaemonSet is deprecated by apps/v1beta2/DaemonSet. See the release notes for
// more information.
// DaemonSet represents the configuration of a daemon set.
type DaemonSet struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// The desired behavior of this daemon set.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Spec DaemonSetSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// The current status of this daemon set. This data may be
	// out of date by some window of time.
	// Populated by the system.
	// Read-only.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Status DaemonSetStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

const (
	// DEPRECATED: DefaultDaemonSetUniqueLabelKey is used instead.
	// DaemonSetTemplateGenerationKey is the key of the labels that is added
	// to daemon set pods to distinguish between old and new pod templates
	// during DaemonSet template update.
	DaemonSetTemplateGenerationKey string = "pod-template-generation"

	// DefaultDaemonSetUniqueLabelKey is the default label key that is added
	// to existing DaemonSet pods to distinguish between old and new
	// DaemonSet pods during DaemonSet template updates.
	DefaultDaemonSetUniqueLabelKey = appsv1beta1.ControllerRevisionHashLabelKey
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DaemonSetList is a collection of daemon sets.
type DaemonSetList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// A list of daemon sets.
	Items []DaemonSet `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Ingress is a collection of rules that allow inbound connections to reach the
// endpoints defined by a backend. An Ingress can be configured to give services
// externally-reachable urls, load balance traffic, terminate SSL, offer name
// based virtual hosting etc.
// DEPRECATED - This group version of Ingress is deprecated by networking.k8s.io/v1beta1 Ingress. See the release notes for more information.
type Ingress struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec is the desired state of the Ingress.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Spec IngressSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// Status is the current state of the Ingress.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Status IngressStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// IngressList is a collection of Ingress.
type IngressList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of Ingress.
	Items []Ingress `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// IngressSpec describes the Ingress the user wishes to exist.
type IngressSpec struct {
	// A default backend capable of servicing requests that don't match any
	// rule. At least one of 'backend' or 'rules' must be specified. This field
	// is optional to allow the loadbalancer controller or defaulting logic to
	// specify a global default.
	// +optional
	Backend *IngressBackend `json:"backend,omitempty" protobuf:"bytes,1,opt,name=backend"`

	// TLS configuration. Currently the Ingress only supports a single TLS
	// port, 443. If multiple members of this list specify different hosts, they
	// will be multiplexed on the same port according to the hostname specified
	// through the SNI TLS extension, if the ingress controller fulfilling the
	// ingress supports SNI.
	// +optional
	TLS []IngressTLS `json:"tls,omitempty" protobuf:"bytes,2,rep,name=tls"`

	// A list of host rules used to configure the Ingress. If unspecified, or
	// no rule matches, all traffic is sent to the default backend.
	// +optional
	Rules []IngressRule `json:"rules,omitempty" protobuf:"bytes,3,rep,name=rules"`
	// TODO: Add the ability to specify load-balancer IP through claims
}

// IngressTLS describes the transport layer security associated with an Ingress.
type IngressTLS struct {
	// Hosts are a list of hosts included in the TLS certificate. The values in
	// this list must match the name/s used in the tlsSecret. Defaults to the
	// wildcard host setting for the loadbalancer controller fulfilling this
	// Ingress, if left unspecified.
	// +optional
	Hosts []string `json:"hosts,omitempty" protobuf:"bytes,1,rep,name=hosts"`
	// SecretName is the name of the secret used to terminate SSL traffic on 443.
	// Field is left optional to allow SSL routing based on SNI hostname alone.
	// If the SNI host in a listener conflicts with the "Host" header field used
	// by an IngressRule, the SNI host is used for termination and value of the
	// Host header is used for routing.
	// +optional
	SecretName string `json:"secretName,omitempty" protobuf:"bytes,2,opt,name=secretName"`
	// TODO: Consider specifying different modes of termination, protocols etc.
}

// IngressStatus describe the current state of the Ingress.
type IngressStatus struct {
	// LoadBalancer contains the current status of the load-balancer.
	// +optional
	LoadBalancer v1.LoadBalancerStatus `json:"loadBalancer,omitempty" protobuf:"bytes,1,opt,name=loadBalancer"`
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
	// +optional
	Host string `json:"host,omitempty" protobuf:"bytes,1,opt,name=host"`
	// IngressRuleValue represents a rule to route requests for this IngressRule.
	// If unspecified, the rule defaults to a http catch-all. Whether that sends
	// just traffic matching the host to the default backend or all traffic to the
	// default backend, is left to the controller fulfilling the Ingress. Http is
	// currently the only supported IngressRuleValue.
	// +optional
	IngressRuleValue `json:",inline,omitempty" protobuf:"bytes,2,opt,name=ingressRuleValue"`
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

	// +optional
	HTTP *HTTPIngressRuleValue `json:"http,omitempty" protobuf:"bytes,1,opt,name=http"`
}

// HTTPIngressRuleValue is a list of http selectors pointing to backends.
// In the example: http://<host>/<path>?<searchpart> -> backend where
// where parts of the url correspond to RFC 3986, this resource will be used
// to match against everything after the last '/' and before the first '?'
// or '#'.
type HTTPIngressRuleValue struct {
	// A collection of paths that map requests to backends.
	Paths []HTTPIngressPath `json:"paths" protobuf:"bytes,1,rep,name=paths"`
	// TODO: Consider adding fields for ingress-type specific global
	// options usable by a loadbalancer, like http keep-alive.
}

// HTTPIngressPath associates a path regex with a backend. Incoming urls matching
// the path are forwarded to the backend.
type HTTPIngressPath struct {
	// Path is an extended POSIX regex as defined by IEEE Std 1003.1,
	// (i.e this follows the egrep/unix syntax, not the perl syntax)
	// matched against the path of an incoming request. Currently it can
	// contain characters disallowed from the conventional "path"
	// part of a URL as defined by RFC 3986. Paths must begin with
	// a '/'. If unspecified, the path defaults to a catch all sending
	// traffic to the backend.
	// +optional
	Path string `json:"path,omitempty" protobuf:"bytes,1,opt,name=path"`

	// Backend defines the referenced service endpoint to which the traffic
	// will be forwarded to.
	Backend IngressBackend `json:"backend" protobuf:"bytes,2,opt,name=backend"`
}

// IngressBackend describes all endpoints for a given service and port.
type IngressBackend struct {
	// Specifies the name of the referenced service.
	ServiceName string `json:"serviceName" protobuf:"bytes,1,opt,name=serviceName"`

	// Specifies the port of the referenced service.
	ServicePort intstr.IntOrString `json:"servicePort" protobuf:"bytes,2,opt,name=servicePort"`
}

// +genclient
// +genclient:method=GetScale,verb=get,subresource=scale,result=Scale
// +genclient:method=UpdateScale,verb=update,subresource=scale,input=Scale,result=Scale
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DEPRECATED - This group version of ReplicaSet is deprecated by apps/v1beta2/ReplicaSet. See the release notes for
// more information.
// ReplicaSet ensures that a specified number of pod replicas are running at any given time.
type ReplicaSet struct {
	metav1.TypeMeta `json:",inline"`

	// If the Labels of a ReplicaSet are empty, they are defaulted to
	// be the same as the Pod(s) that the ReplicaSet manages.
	// Standard object's metadata. More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec defines the specification of the desired behavior of the ReplicaSet.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Spec ReplicaSetSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// Status is the most recently observed status of the ReplicaSet.
	// This data may be out of date by some window of time.
	// Populated by the system.
	// Read-only.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Status ReplicaSetStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ReplicaSetList is a collection of ReplicaSets.
type ReplicaSetList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// List of ReplicaSets.
	// More info: https://kubernetes.io/docs/concepts/workloads/controllers/replicationcontroller
	Items []ReplicaSet `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// ReplicaSetSpec is the specification of a ReplicaSet.
type ReplicaSetSpec struct {
	// Replicas is the number of desired replicas.
	// This is a pointer to distinguish between explicit zero and unspecified.
	// Defaults to 1.
	// More info: https://kubernetes.io/docs/concepts/workloads/controllers/replicationcontroller/#what-is-a-replicationcontroller
	// +optional
	Replicas *int32 `json:"replicas,omitempty" protobuf:"varint,1,opt,name=replicas"`

	// Minimum number of seconds for which a newly created pod should be ready
	// without any of its container crashing, for it to be considered available.
	// Defaults to 0 (pod will be considered available as soon as it is ready)
	// +optional
	MinReadySeconds int32 `json:"minReadySeconds,omitempty" protobuf:"varint,4,opt,name=minReadySeconds"`

	// Selector is a label query over pods that should match the replica count.
	// If the selector is empty, it is defaulted to the labels present on the pod template.
	// Label keys and values that must match in order to be controlled by this replica set.
	// More info: https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#label-selectors
	// +optional
	Selector *metav1.LabelSelector `json:"selector,omitempty" protobuf:"bytes,2,opt,name=selector"`

	// Template is the object that describes the pod that will be created if
	// insufficient replicas are detected.
	// More info: https://kubernetes.io/docs/concepts/workloads/controllers/replicationcontroller#pod-template
	// +optional
	Template v1.PodTemplateSpec `json:"template,omitempty" protobuf:"bytes,3,opt,name=template"`
}

// ReplicaSetStatus represents the current status of a ReplicaSet.
type ReplicaSetStatus struct {
	// Replicas is the most recently oberved number of replicas.
	// More info: https://kubernetes.io/docs/concepts/workloads/controllers/replicationcontroller/#what-is-a-replicationcontroller
	Replicas int32 `json:"replicas" protobuf:"varint,1,opt,name=replicas"`

	// The number of pods that have labels matching the labels of the pod template of the replicaset.
	// +optional
	FullyLabeledReplicas int32 `json:"fullyLabeledReplicas,omitempty" protobuf:"varint,2,opt,name=fullyLabeledReplicas"`

	// The number of ready replicas for this replica set.
	// +optional
	ReadyReplicas int32 `json:"readyReplicas,omitempty" protobuf:"varint,4,opt,name=readyReplicas"`

	// The number of available replicas (ready for at least minReadySeconds) for this replica set.
	// +optional
	AvailableReplicas int32 `json:"availableReplicas,omitempty" protobuf:"varint,5,opt,name=availableReplicas"`

	// ObservedGeneration reflects the generation of the most recently observed ReplicaSet.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty" protobuf:"varint,3,opt,name=observedGeneration"`

	// Represents the latest available observations of a replica set's current state.
	// +optional
	// +patchMergeKey=type
	// +patchStrategy=merge
	Conditions []ReplicaSetCondition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,6,rep,name=conditions"`
}

type ReplicaSetConditionType string

// These are valid conditions of a replica set.
const (
	// ReplicaSetReplicaFailure is added in a replica set when one of its pods fails to be created
	// due to insufficient quota, limit ranges, pod security policy, node selectors, etc. or deleted
	// due to kubelet being down or finalizers are failing.
	ReplicaSetReplicaFailure ReplicaSetConditionType = "ReplicaFailure"
)

// ReplicaSetCondition describes the state of a replica set at a certain point.
type ReplicaSetCondition struct {
	// Type of replica set condition.
	Type ReplicaSetConditionType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=ReplicaSetConditionType"`
	// Status of the condition, one of True, False, Unknown.
	Status v1.ConditionStatus `json:"status" protobuf:"bytes,2,opt,name=status,casttype=k8s.io/api/core/v1.ConditionStatus"`
	// The last time the condition transitioned from one status to another.
	// +optional
	LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty" protobuf:"bytes,3,opt,name=lastTransitionTime"`
	// The reason for the condition's last transition.
	// +optional
	Reason string `json:"reason,omitempty" protobuf:"bytes,4,opt,name=reason"`
	// A human readable message indicating details about the transition.
	// +optional
	Message string `json:"message,omitempty" protobuf:"bytes,5,opt,name=message"`
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodSecurityPolicy governs the ability to make requests that affect the Security Context
// that will be applied to a pod and container.
// Deprecated: use PodSecurityPolicy from policy API Group instead.
type PodSecurityPolicy struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec defines the policy enforced.
	// +optional
	Spec PodSecurityPolicySpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
}

// PodSecurityPolicySpec defines the policy enforced.
// Deprecated: use PodSecurityPolicySpec from policy API Group instead.
type PodSecurityPolicySpec struct {
	// privileged determines if a pod can request to be run as privileged.
	// +optional
	Privileged bool `json:"privileged,omitempty" protobuf:"varint,1,opt,name=privileged"`
	// defaultAddCapabilities is the default set of capabilities that will be added to the container
	// unless the pod spec specifically drops the capability.  You may not list a capability in both
	// defaultAddCapabilities and requiredDropCapabilities. Capabilities added here are implicitly
	// allowed, and need not be included in the allowedCapabilities list.
	// +optional
	DefaultAddCapabilities []v1.Capability `json:"defaultAddCapabilities,omitempty" protobuf:"bytes,2,rep,name=defaultAddCapabilities,casttype=k8s.io/api/core/v1.Capability"`
	// requiredDropCapabilities are the capabilities that will be dropped from the container.  These
	// are required to be dropped and cannot be added.
	// +optional
	RequiredDropCapabilities []v1.Capability `json:"requiredDropCapabilities,omitempty" protobuf:"bytes,3,rep,name=requiredDropCapabilities,casttype=k8s.io/api/core/v1.Capability"`
	// allowedCapabilities is a list of capabilities that can be requested to add to the container.
	// Capabilities in this field may be added at the pod author's discretion.
	// You must not list a capability in both allowedCapabilities and requiredDropCapabilities.
	// +optional
	AllowedCapabilities []v1.Capability `json:"allowedCapabilities,omitempty" protobuf:"bytes,4,rep,name=allowedCapabilities,casttype=k8s.io/api/core/v1.Capability"`
	// volumes is a white list of allowed volume plugins. Empty indicates that
	// no volumes may be used. To allow all volumes you may use '*'.
	// +optional
	Volumes []FSType `json:"volumes,omitempty" protobuf:"bytes,5,rep,name=volumes,casttype=FSType"`
	// hostNetwork determines if the policy allows the use of HostNetwork in the pod spec.
	// +optional
	HostNetwork bool `json:"hostNetwork,omitempty" protobuf:"varint,6,opt,name=hostNetwork"`
	// hostPorts determines which host port ranges are allowed to be exposed.
	// +optional
	HostPorts []HostPortRange `json:"hostPorts,omitempty" protobuf:"bytes,7,rep,name=hostPorts"`
	// hostPID determines if the policy allows the use of HostPID in the pod spec.
	// +optional
	HostPID bool `json:"hostPID,omitempty" protobuf:"varint,8,opt,name=hostPID"`
	// hostIPC determines if the policy allows the use of HostIPC in the pod spec.
	// +optional
	HostIPC bool `json:"hostIPC,omitempty" protobuf:"varint,9,opt,name=hostIPC"`
	// seLinux is the strategy that will dictate the allowable labels that may be set.
	SELinux SELinuxStrategyOptions `json:"seLinux" protobuf:"bytes,10,opt,name=seLinux"`
	// runAsUser is the strategy that will dictate the allowable RunAsUser values that may be set.
	RunAsUser RunAsUserStrategyOptions `json:"runAsUser" protobuf:"bytes,11,opt,name=runAsUser"`
	// RunAsGroup is the strategy that will dictate the allowable RunAsGroup values that may be set.
	// If this field is omitted, the pod's RunAsGroup can take any value. This field requires the
	// RunAsGroup feature gate to be enabled.
	// +optional
	RunAsGroup *RunAsGroupStrategyOptions `json:"runAsGroup,omitempty" protobuf:"bytes,22,opt,name=runAsGroup"`
	// supplementalGroups is the strategy that will dictate what supplemental groups are used by the SecurityContext.
	SupplementalGroups SupplementalGroupsStrategyOptions `json:"supplementalGroups" protobuf:"bytes,12,opt,name=supplementalGroups"`
	// fsGroup is the strategy that will dictate what fs group is used by the SecurityContext.
	FSGroup FSGroupStrategyOptions `json:"fsGroup" protobuf:"bytes,13,opt,name=fsGroup"`
	// readOnlyRootFilesystem when set to true will force containers to run with a read only root file
	// system.  If the container specifically requests to run with a non-read only root file system
	// the PSP should deny the pod.
	// If set to false the container may run with a read only root file system if it wishes but it
	// will not be forced to.
	// +optional
	ReadOnlyRootFilesystem bool `json:"readOnlyRootFilesystem,omitempty" protobuf:"varint,14,opt,name=readOnlyRootFilesystem"`
	// defaultAllowPrivilegeEscalation controls the default setting for whether a
	// process can gain more privileges than its parent process.
	// +optional
	DefaultAllowPrivilegeEscalation *bool `json:"defaultAllowPrivilegeEscalation,omitempty" protobuf:"varint,15,opt,name=defaultAllowPrivilegeEscalation"`
	// allowPrivilegeEscalation determines if a pod can request to allow
	// privilege escalation. If unspecified, defaults to true.
	// +optional
	AllowPrivilegeEscalation *bool `json:"allowPrivilegeEscalation,omitempty" protobuf:"varint,16,opt,name=allowPrivilegeEscalation"`
	// allowedHostPaths is a white list of allowed host paths. Empty indicates
	// that all host paths may be used.
	// +optional
	AllowedHostPaths []AllowedHostPath `json:"allowedHostPaths,omitempty" protobuf:"bytes,17,rep,name=allowedHostPaths"`
	// allowedFlexVolumes is a whitelist of allowed Flexvolumes.  Empty or nil indicates that all
	// Flexvolumes may be used.  This parameter is effective only when the usage of the Flexvolumes
	// is allowed in the "volumes" field.
	// +optional
	AllowedFlexVolumes []AllowedFlexVolume `json:"allowedFlexVolumes,omitempty" protobuf:"bytes,18,rep,name=allowedFlexVolumes"`
	// AllowedCSIDrivers is a whitelist of inline CSI drivers that must be explicitly set to be embedded within a pod spec.
	// An empty value means no CSI drivers can run inline within a pod spec.
	// +optional
	AllowedCSIDrivers []AllowedCSIDriver `json:"allowedCSIDrivers,omitempty" protobuf:"bytes,23,rep,name=allowedCSIDrivers"`
	// allowedUnsafeSysctls is a list of explicitly allowed unsafe sysctls, defaults to none.
	// Each entry is either a plain sysctl name or ends in "*" in which case it is considered
	// as a prefix of allowed sysctls. Single * means all unsafe sysctls are allowed.
	// Kubelet has to whitelist all allowed unsafe sysctls explicitly to avoid rejection.
	//
	// Examples:
	// e.g. "foo/*" allows "foo/bar", "foo/baz", etc.
	// e.g. "foo.*" allows "foo.bar", "foo.baz", etc.
	// +optional
	AllowedUnsafeSysctls []string `json:"allowedUnsafeSysctls,omitempty" protobuf:"bytes,19,rep,name=allowedUnsafeSysctls"`
	// forbiddenSysctls is a list of explicitly forbidden sysctls, defaults to none.
	// Each entry is either a plain sysctl name or ends in "*" in which case it is considered
	// as a prefix of forbidden sysctls. Single * means all sysctls are forbidden.
	//
	// Examples:
	// e.g. "foo/*" forbids "foo/bar", "foo/baz", etc.
	// e.g. "foo.*" forbids "foo.bar", "foo.baz", etc.
	// +optional
	ForbiddenSysctls []string `json:"forbiddenSysctls,omitempty" protobuf:"bytes,20,rep,name=forbiddenSysctls"`
	// AllowedProcMountTypes is a whitelist of allowed ProcMountTypes.
	// Empty or nil indicates that only the DefaultProcMountType may be used.
	// This requires the ProcMountType feature flag to be enabled.
	// +optional
	AllowedProcMountTypes []v1.ProcMountType `json:"allowedProcMountTypes,omitempty" protobuf:"bytes,21,opt,name=allowedProcMountTypes"`
}

// AllowedHostPath defines the host volume conditions that will be enabled by a policy
// for pods to use. It requires the path prefix to be defined.
// Deprecated: use AllowedHostPath from policy API Group instead.
type AllowedHostPath struct {
	// pathPrefix is the path prefix that the host volume must match.
	// It does not support `*`.
	// Trailing slashes are trimmed when validating the path prefix with a host path.
	//
	// Examples:
	// `/foo` would allow `/foo`, `/foo/` and `/foo/bar`
	// `/foo` would not allow `/food` or `/etc/foo`
	PathPrefix string `json:"pathPrefix,omitempty" protobuf:"bytes,1,rep,name=pathPrefix"`

	// when set to true, will allow host volumes matching the pathPrefix only if all volume mounts are readOnly.
	// +optional
	ReadOnly bool `json:"readOnly,omitempty" protobuf:"varint,2,opt,name=readOnly"`
}

// FSType gives strong typing to different file systems that are used by volumes.
// Deprecated: use FSType from policy API Group instead.
type FSType string

var (
	AzureFile             FSType = "azureFile"
	Flocker               FSType = "flocker"
	FlexVolume            FSType = "flexVolume"
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
	ConfigMap             FSType = "configMap"
	Quobyte               FSType = "quobyte"
	AzureDisk             FSType = "azureDisk"
	CSI                   FSType = "csi"
	All                   FSType = "*"
)

// AllowedFlexVolume represents a single Flexvolume that is allowed to be used.
// Deprecated: use AllowedFlexVolume from policy API Group instead.
type AllowedFlexVolume struct {
	// driver is the name of the Flexvolume driver.
	Driver string `json:"driver" protobuf:"bytes,1,opt,name=driver"`
}

// AllowedCSIDriver represents a single inline CSI Driver that is allowed to be used.
type AllowedCSIDriver struct {
	// Name is the registered name of the CSI driver
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`
}

// HostPortRange defines a range of host ports that will be enabled by a policy
// for pods to use.  It requires both the start and end to be defined.
// Deprecated: use HostPortRange from policy API Group instead.
type HostPortRange struct {
	// min is the start of the range, inclusive.
	Min int32 `json:"min" protobuf:"varint,1,opt,name=min"`
	// max is the end of the range, inclusive.
	Max int32 `json:"max" protobuf:"varint,2,opt,name=max"`
}

// SELinuxStrategyOptions defines the strategy type and any options used to create the strategy.
// Deprecated: use SELinuxStrategyOptions from policy API Group instead.
type SELinuxStrategyOptions struct {
	// rule is the strategy that will dictate the allowable labels that may be set.
	Rule SELinuxStrategy `json:"rule" protobuf:"bytes,1,opt,name=rule,casttype=SELinuxStrategy"`
	// seLinuxOptions required to run as; required for MustRunAs
	// More info: https://kubernetes.io/docs/tasks/configure-pod-container/security-context/
	// +optional
	SELinuxOptions *v1.SELinuxOptions `json:"seLinuxOptions,omitempty" protobuf:"bytes,2,opt,name=seLinuxOptions"`
}

// SELinuxStrategy denotes strategy types for generating SELinux options for a
// Security Context.
// Deprecated: use SELinuxStrategy from policy API Group instead.
type SELinuxStrategy string

const (
	// SELinuxStrategyMustRunAs means that container must have SELinux labels of X applied.
	// Deprecated: use SELinuxStrategyMustRunAs from policy API Group instead.
	SELinuxStrategyMustRunAs SELinuxStrategy = "MustRunAs"
	// SELinuxStrategyRunAsAny means that container may make requests for any SELinux context labels.
	// Deprecated: use SELinuxStrategyRunAsAny from policy API Group instead.
	SELinuxStrategyRunAsAny SELinuxStrategy = "RunAsAny"
)

// RunAsUserStrategyOptions defines the strategy type and any options used to create the strategy.
// Deprecated: use RunAsUserStrategyOptions from policy API Group instead.
type RunAsUserStrategyOptions struct {
	// rule is the strategy that will dictate the allowable RunAsUser values that may be set.
	Rule RunAsUserStrategy `json:"rule" protobuf:"bytes,1,opt,name=rule,casttype=RunAsUserStrategy"`
	// ranges are the allowed ranges of uids that may be used. If you would like to force a single uid
	// then supply a single range with the same start and end. Required for MustRunAs.
	// +optional
	Ranges []IDRange `json:"ranges,omitempty" protobuf:"bytes,2,rep,name=ranges"`
}

// RunAsGroupStrategyOptions defines the strategy type and any options used to create the strategy.
// Deprecated: use RunAsGroupStrategyOptions from policy API Group instead.
type RunAsGroupStrategyOptions struct {
	// rule is the strategy that will dictate the allowable RunAsGroup values that may be set.
	Rule RunAsGroupStrategy `json:"rule" protobuf:"bytes,1,opt,name=rule,casttype=RunAsGroupStrategy"`
	// ranges are the allowed ranges of gids that may be used. If you would like to force a single gid
	// then supply a single range with the same start and end. Required for MustRunAs.
	// +optional
	Ranges []IDRange `json:"ranges,omitempty" protobuf:"bytes,2,rep,name=ranges"`
}

// IDRange provides a min/max of an allowed range of IDs.
// Deprecated: use IDRange from policy API Group instead.
type IDRange struct {
	// min is the start of the range, inclusive.
	Min int64 `json:"min" protobuf:"varint,1,opt,name=min"`
	// max is the end of the range, inclusive.
	Max int64 `json:"max" protobuf:"varint,2,opt,name=max"`
}

// RunAsUserStrategy denotes strategy types for generating RunAsUser values for a
// Security Context.
// Deprecated: use RunAsUserStrategy from policy API Group instead.
type RunAsUserStrategy string

const (
	// RunAsUserStrategyMustRunAs means that container must run as a particular uid.
	// Deprecated: use RunAsUserStrategyMustRunAs from policy API Group instead.
	RunAsUserStrategyMustRunAs RunAsUserStrategy = "MustRunAs"
	// RunAsUserStrategyMustRunAsNonRoot means that container must run as a non-root uid.
	// Deprecated: use RunAsUserStrategyMustRunAsNonRoot from policy API Group instead.
	RunAsUserStrategyMustRunAsNonRoot RunAsUserStrategy = "MustRunAsNonRoot"
	// RunAsUserStrategyRunAsAny means that container may make requests for any uid.
	// Deprecated: use RunAsUserStrategyRunAsAny from policy API Group instead.
	RunAsUserStrategyRunAsAny RunAsUserStrategy = "RunAsAny"
)

// RunAsGroupStrategy denotes strategy types for generating RunAsGroup values for a
// Security Context.
// Deprecated: use RunAsGroupStrategy from policy API Group instead.
type RunAsGroupStrategy string

const (
	// RunAsGroupStrategyMayRunAs means that container does not need to run with a particular gid.
	// However, when RunAsGroup are specified, they have to fall in the defined range.
	RunAsGroupStrategyMayRunAs RunAsGroupStrategy = "MayRunAs"
	// RunAsGroupStrategyMustRunAs means that container must run as a particular gid.
	// Deprecated: use RunAsGroupStrategyMustRunAs from policy API Group instead.
	RunAsGroupStrategyMustRunAs RunAsGroupStrategy = "MustRunAs"
	// RunAsGroupStrategyRunAsAny means that container may make requests for any gid.
	// Deprecated: use RunAsGroupStrategyRunAsAny from policy API Group instead.
	RunAsGroupStrategyRunAsAny RunAsGroupStrategy = "RunAsAny"
)

// FSGroupStrategyOptions defines the strategy type and options used to create the strategy.
// Deprecated: use FSGroupStrategyOptions from policy API Group instead.
type FSGroupStrategyOptions struct {
	// rule is the strategy that will dictate what FSGroup is used in the SecurityContext.
	// +optional
	Rule FSGroupStrategyType `json:"rule,omitempty" protobuf:"bytes,1,opt,name=rule,casttype=FSGroupStrategyType"`
	// ranges are the allowed ranges of fs groups.  If you would like to force a single
	// fs group then supply a single range with the same start and end. Required for MustRunAs.
	// +optional
	Ranges []IDRange `json:"ranges,omitempty" protobuf:"bytes,2,rep,name=ranges"`
}

// FSGroupStrategyType denotes strategy types for generating FSGroup values for a
// SecurityContext
// Deprecated: use FSGroupStrategyType from policy API Group instead.
type FSGroupStrategyType string

const (
	// FSGroupStrategyMustRunAs meant that container must have FSGroup of X applied.
	// Deprecated: use FSGroupStrategyMustRunAs from policy API Group instead.
	FSGroupStrategyMustRunAs FSGroupStrategyType = "MustRunAs"
	// FSGroupStrategyRunAsAny means that container may make requests for any FSGroup labels.
	// Deprecated: use FSGroupStrategyRunAsAny from policy API Group instead.
	FSGroupStrategyRunAsAny FSGroupStrategyType = "RunAsAny"
)

// SupplementalGroupsStrategyOptions defines the strategy type and options used to create the strategy.
// Deprecated: use SupplementalGroupsStrategyOptions from policy API Group instead.
type SupplementalGroupsStrategyOptions struct {
	// rule is the strategy that will dictate what supplemental groups is used in the SecurityContext.
	// +optional
	Rule SupplementalGroupsStrategyType `json:"rule,omitempty" protobuf:"bytes,1,opt,name=rule,casttype=SupplementalGroupsStrategyType"`
	// ranges are the allowed ranges of supplemental groups.  If you would like to force a single
	// supplemental group then supply a single range with the same start and end. Required for MustRunAs.
	// +optional
	Ranges []IDRange `json:"ranges,omitempty" protobuf:"bytes,2,rep,name=ranges"`
}

// SupplementalGroupsStrategyType denotes strategy types for determining valid supplemental
// groups for a SecurityContext.
// Deprecated: use SupplementalGroupsStrategyType from policy API Group instead.
type SupplementalGroupsStrategyType string

const (
	// SupplementalGroupsStrategyMustRunAs means that container must run as a particular gid.
	// Deprecated: use SupplementalGroupsStrategyMustRunAs from policy API Group instead.
	SupplementalGroupsStrategyMustRunAs SupplementalGroupsStrategyType = "MustRunAs"
	// SupplementalGroupsStrategyRunAsAny means that container may make requests for any gid.
	// Deprecated: use SupplementalGroupsStrategyRunAsAny from policy API Group instead.
	SupplementalGroupsStrategyRunAsAny SupplementalGroupsStrategyType = "RunAsAny"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodSecurityPolicyList is a list of PodSecurityPolicy objects.
// Deprecated: use PodSecurityPolicyList from policy API Group instead.
type PodSecurityPolicyList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is a list of schema objects.
	Items []PodSecurityPolicy `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DEPRECATED 1.9 - This group version of NetworkPolicy is deprecated by networking/v1/NetworkPolicy.
// NetworkPolicy describes what network traffic is allowed for a set of Pods
type NetworkPolicy struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Specification of the desired behavior for this NetworkPolicy.
	// +optional
	Spec NetworkPolicySpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
}

// DEPRECATED 1.9 - This group version of PolicyType is deprecated by networking/v1/PolicyType.
// Policy Type string describes the NetworkPolicy type
// This type is beta-level in 1.8
type PolicyType string

const (
	// PolicyTypeIngress is a NetworkPolicy that affects ingress traffic on selected pods
	PolicyTypeIngress PolicyType = "Ingress"
	// PolicyTypeEgress is a NetworkPolicy that affects egress traffic on selected pods
	PolicyTypeEgress PolicyType = "Egress"
)

// DEPRECATED 1.9 - This group version of NetworkPolicySpec is deprecated by networking/v1/NetworkPolicySpec.
type NetworkPolicySpec struct {
	// Selects the pods to which this NetworkPolicy object applies.  The array of ingress rules
	// is applied to any pods selected by this field. Multiple network policies can select the
	// same set of pods.  In this case, the ingress rules for each are combined additively.
	// This field is NOT optional and follows standard label selector semantics.
	// An empty podSelector matches all pods in this namespace.
	PodSelector metav1.LabelSelector `json:"podSelector" protobuf:"bytes,1,opt,name=podSelector"`

	// List of ingress rules to be applied to the selected pods.
	// Traffic is allowed to a pod if there are no NetworkPolicies selecting the pod
	// OR if the traffic source is the pod's local node,
	// OR if the traffic matches at least one ingress rule across all of the NetworkPolicy
	// objects whose podSelector matches the pod.
	// If this field is empty then this NetworkPolicy does not allow any traffic
	// (and serves solely to ensure that the pods it selects are isolated by default).
	// +optional
	Ingress []NetworkPolicyIngressRule `json:"ingress,omitempty" protobuf:"bytes,2,rep,name=ingress"`

	// List of egress rules to be applied to the selected pods. Outgoing traffic is
	// allowed if there are no NetworkPolicies selecting the pod (and cluster policy
	// otherwise allows the traffic), OR if the traffic matches at least one egress rule
	// across all of the NetworkPolicy objects whose podSelector matches the pod. If
	// this field is empty then this NetworkPolicy limits all outgoing traffic (and serves
	// solely to ensure that the pods it selects are isolated by default).
	// This field is beta-level in 1.8
	// +optional
	Egress []NetworkPolicyEgressRule `json:"egress,omitempty" protobuf:"bytes,3,rep,name=egress"`

	// List of rule types that the NetworkPolicy relates to.
	// Valid options are "Ingress", "Egress", or "Ingress,Egress".
	// If this field is not specified, it will default based on the existence of Ingress or Egress rules;
	// policies that contain an Egress section are assumed to affect Egress, and all policies
	// (whether or not they contain an Ingress section) are assumed to affect Ingress.
	// If you want to write an egress-only policy, you must explicitly specify policyTypes [ "Egress" ].
	// Likewise, if you want to write a policy that specifies that no egress is allowed,
	// you must specify a policyTypes value that include "Egress" (since such a policy would not include
	// an Egress section and would otherwise default to just [ "Ingress" ]).
	// This field is beta-level in 1.8
	// +optional
	PolicyTypes []PolicyType `json:"policyTypes,omitempty" protobuf:"bytes,4,rep,name=policyTypes,casttype=PolicyType"`
}

// DEPRECATED 1.9 - This group version of NetworkPolicyIngressRule is deprecated by networking/v1/NetworkPolicyIngressRule.
// This NetworkPolicyIngressRule matches traffic if and only if the traffic matches both ports AND from.
type NetworkPolicyIngressRule struct {
	// List of ports which should be made accessible on the pods selected for this rule.
	// Each item in this list is combined using a logical OR.
	// If this field is empty or missing, this rule matches all ports (traffic not restricted by port).
	// If this field is present and contains at least one item, then this rule allows traffic
	// only if the traffic matches at least one port in the list.
	// +optional
	Ports []NetworkPolicyPort `json:"ports,omitempty" protobuf:"bytes,1,rep,name=ports"`

	// List of sources which should be able to access the pods selected for this rule.
	// Items in this list are combined using a logical OR operation.
	// If this field is empty or missing, this rule matches all sources (traffic not restricted by source).
	// If this field is present and contains at least on item, this rule allows traffic only if the
	// traffic matches at least one item in the from list.
	// +optional
	From []NetworkPolicyPeer `json:"from,omitempty" protobuf:"bytes,2,rep,name=from"`
}

// DEPRECATED 1.9 - This group version of NetworkPolicyEgressRule is deprecated by networking/v1/NetworkPolicyEgressRule.
// NetworkPolicyEgressRule describes a particular set of traffic that is allowed out of pods
// matched by a NetworkPolicySpec's podSelector. The traffic must match both ports and to.
// This type is beta-level in 1.8
type NetworkPolicyEgressRule struct {
	// List of destination ports for outgoing traffic.
	// Each item in this list is combined using a logical OR. If this field is
	// empty or missing, this rule matches all ports (traffic not restricted by port).
	// If this field is present and contains at least one item, then this rule allows
	// traffic only if the traffic matches at least one port in the list.
	// +optional
	Ports []NetworkPolicyPort `json:"ports,omitempty" protobuf:"bytes,1,rep,name=ports"`

	// List of destinations for outgoing traffic of pods selected for this rule.
	// Items in this list are combined using a logical OR operation. If this field is
	// empty or missing, this rule matches all destinations (traffic not restricted by
	// destination). If this field is present and contains at least one item, this rule
	// allows traffic only if the traffic matches at least one item in the to list.
	// +optional
	To []NetworkPolicyPeer `json:"to,omitempty" protobuf:"bytes,2,rep,name=to"`
}

// DEPRECATED 1.9 - This group version of NetworkPolicyPort is deprecated by networking/v1/NetworkPolicyPort.
type NetworkPolicyPort struct {
	// Optional.  The protocol (TCP, UDP, or SCTP) which traffic must match.
	// If not specified, this field defaults to TCP.
	// +optional
	Protocol *v1.Protocol `json:"protocol,omitempty" protobuf:"bytes,1,opt,name=protocol,casttype=k8s.io/api/core/v1.Protocol"`

	// If specified, the port on the given protocol.  This can
	// either be a numerical or named port on a pod.  If this field is not provided,
	// this matches all port names and numbers.
	// If present, only traffic on the specified protocol AND port
	// will be matched.
	// +optional
	Port *intstr.IntOrString `json:"port,omitempty" protobuf:"bytes,2,opt,name=port"`
}

// DEPRECATED 1.9 - This group version of IPBlock is deprecated by networking/v1/IPBlock.
// IPBlock describes a particular CIDR (Ex. "192.168.1.1/24") that is allowed to the pods
// matched by a NetworkPolicySpec's podSelector. The except entry describes CIDRs that should
// not be included within this rule.
type IPBlock struct {
	// CIDR is a string representing the IP Block
	// Valid examples are "192.168.1.1/24"
	CIDR string `json:"cidr" protobuf:"bytes,1,name=cidr"`
	// Except is a slice of CIDRs that should not be included within an IP Block
	// Valid examples are "192.168.1.1/24"
	// Except values will be rejected if they are outside the CIDR range
	// +optional
	Except []string `json:"except,omitempty" protobuf:"bytes,2,rep,name=except"`
}

// DEPRECATED 1.9 - This group version of NetworkPolicyPeer is deprecated by networking/v1/NetworkPolicyPeer.
type NetworkPolicyPeer struct {
	// This is a label selector which selects Pods. This field follows standard label
	// selector semantics; if present but empty, it selects all pods.
	//
	// If NamespaceSelector is also set, then the NetworkPolicyPeer as a whole selects
	// the Pods matching PodSelector in the Namespaces selected by NamespaceSelector.
	// Otherwise it selects the Pods matching PodSelector in the policy's own Namespace.
	// +optional
	PodSelector *metav1.LabelSelector `json:"podSelector,omitempty" protobuf:"bytes,1,opt,name=podSelector"`

	// Selects Namespaces using cluster-scoped labels. This field follows standard label
	// selector semantics; if present but empty, it selects all namespaces.
	//
	// If PodSelector is also set, then the NetworkPolicyPeer as a whole selects
	// the Pods matching PodSelector in the Namespaces selected by NamespaceSelector.
	// Otherwise it selects all Pods in the Namespaces selected by NamespaceSelector.
	// +optional
	NamespaceSelector *metav1.LabelSelector `json:"namespaceSelector,omitempty" protobuf:"bytes,2,opt,name=namespaceSelector"`

	// IPBlock defines policy on a particular IPBlock. If this field is set then
	// neither of the other fields can be.
	// +optional
	IPBlock *IPBlock `json:"ipBlock,omitempty" protobuf:"bytes,3,rep,name=ipBlock"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DEPRECATED 1.9 - This group version of NetworkPolicyList is deprecated by networking/v1/NetworkPolicyList.
// Network Policy List is a list of NetworkPolicy objects.
type NetworkPolicyList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is a list of schema objects.
	Items []NetworkPolicy `json:"items" protobuf:"bytes,2,rep,name=items"`
}
