/*
Copyright The Kubernetes Authors.

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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apimachinerytypes "k8s.io/apimachinery/pkg/types"
)

// KubernetesEvictionResponderName is the name identifying a core in-tree responder.
type KubernetesEvictionResponderName string

const (
	// EvictionResponderImperativeEviction is a default responder that will evict pods using the imperative
	// Eviction API (/evict endpoint) with a backoff.
	EvictionResponderImperativeEviction KubernetesEvictionResponderName = "imperative-eviction.k8s.io/evictor"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.36
// +k8s:supportsSubresource="/status"

// EvictionRequest defines a request that should ideally result in a graceful eviction of a
// .spec.target (e.g. termination of a pod).
//
// `.spec.requesters` field should be set and kept updated to preserve the eviction request.
//
// If the target is a pod, the .status.targetResponders is populated from Pod's
// .spec.evictionResponders.
//
// Responders should observe and communicate through the .status to help with the eviction
// of the target when they see their state == Active in .status.targetResponders. ResponderStatus
// struct should then be periodically updated to indicate the progress or completion of the eviction
// process by each responder in .status.responders. If .status.responders[].heartbeatTime is
// not updated within 20 minutes, the eviction request is passed over to the next responder.
//
// If there are no other responders and the target is a pod, the last default
// imperative-eviction.k8s.io/evictor responder will evict the pod using the imperative Eviction API
// (/evict endpoint).
// +k8s:validation-gen-nolint // Note: remove this when the API got GA
type EvictionRequest struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object metadata; More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata.
	// .metadata.name is required and should match the .metadata.uid of the pod being evicted.
	// .metadata.generateName is not supported.
	// The labels of the eviction request object are synchronized with .metadata.labels of the
	// eviction request's target. The labels of the target have a preference.
	// +optional
	// +k8s:subfield(name)=+k8s:format=k8s-uuid
	// +k8s:subfield(generateName)=+k8s:forbidden
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec defines the eviction request specification.
	// https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +required
	Spec EvictionRequestSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// status represents the most recently observed status of the eviction request.
	// Populated by responders and evictionrequest-controller.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Status EvictionRequestStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// EvictionRequestSpec is a specification of an EvictionRequest.
type EvictionRequestSpec struct {
	// target contains a reference to an object (e.g. a pod) that should be evicted.
	// Target UID must be the same as the EvictionRequest's .metadata.name.
	// This field is required and immutable.
	// +required
	// +k8s:immutable
	Target EvictionTarget `json:"target" protobuf:"bytes,1,opt,name=target"`

	// requesters allow you to identify entities, that requested the eviction of the target.
	// At least one requester with the eviction intent is required when creating an eviction request.
	// If all the requesters withdraw their eviction intent, the eviction will be canceled.
	//
	// Requester controllers should use server-side-apply to manage the requester intent.
	//
	// Once added, items cannot be removed.
	// The minimum length of the requesters list is 1, and the maximum is 100.
	// +required
	// +patchMergeKey=name
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=name
	// +k8s:required
	// +k8s:listType=map
	// +k8s:listMapKey=name
	// +k8s:maxItems=100
	Requesters []Requester `json:"requesters,omitempty" patchStrategy:"merge" patchMergeKey:"name" protobuf:"bytes,2,rep,name=requesters"`
}

// EvictionTarget contains a reference to an object that should be evicted.
// +union
type EvictionTarget struct {
	// pod references a pod that is subject to eviction/termination.
	// Pods that are part of a PodGroup (.spec.schedulingGroup is set) are not supported.
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	Pod *PodReference `json:"pod,omitempty" protobuf:"bytes,1,opt,name=pod"`
}

// PodReference contains enough information to locate the referenced pod inside the same namespace.
type PodReference struct {
	// name of the target.
	// This field is required.
	// +required
	// +k8s:required
	// +k8s:format=k8s-long-name
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`
	// uid of the target.
	// It can be found in .spec.metadata.uid of the target and is a lowercase UUID in 8-4-4-4-12 format.
	// This field is required.
	// +required
	// +k8s:required
	// +k8s:format=k8s-uuid
	UID apimachinerytypes.UID `json:"uid" protobuf:"bytes,2,opt,name=uid,casttype=k8s.io/kubernetes/pkg/types.UID"`
}

// Requester allows you to identify the entity, that requested the eviction of the target.
// +structType=atomic
type Requester struct {
	// name allows you to identify the entity, that requested the eviction of the target.
	//
	// It must be a valid domain-prefixed path (such as "acme.io/foo").
	// Domain names *.k8s.io and *.kubernetes.io are reserved.
	// This field must be unique for each requester.
	// This field is required.
	// +required
	// +k8s:required
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`

	// intent specifies the action that should be taken for the specified target.
	//
	// - Eviction means that the requester is interested in the eviction of the target.
	// - Withdrawn means that the requester is no longer interested in the eviction of the target.
	//   If all requesters' intents are withdrawn, the eviction will be canceled.
	//   Cancellation consequences:
	//   - Inactive responders will never run.
	//   - Active responders are expected to cancel the eviction.
	//   - Completed or Interrupted responders should not take any action.
	// +required
	// +k8s:required
	Intent RequesterIntent `json:"intent" protobuf:"bytes,2,opt,name=intent,casttype=RequesterIntentEviction"`
}

// RequesterIntent specifies a requester intent.
// +k8s:enum
type RequesterIntent string

// These are intents that can be set by each requester.
const (
	// RequesterIntentEviction means that the requester is interested in the eviction of the target.
	RequesterIntentEviction RequesterIntent = "Eviction"

	// RequesterIntentWithdrawn means that the requester is no longer interested in the eviction of the target.
	// If all requesters' intents are withdrawn, the eviction will be canceled.
	// Cancellation consequences:
	// - Inactive responders will never run.
	// - Active responders are expected to cancel the eviction.
	// - Completed or Interrupted responders should not take any action.
	RequesterIntentWithdrawn RequesterIntent = "Withdrawn"
)

// EvictionRequestStatus represents the last observed status of the eviction request.
type EvictionRequestStatus struct {
	// conditions contain information about the eviction request.
	//
	// Eviction request specific conditions are: Evicted or Canceled (managed by evictionrequest-controller),
	//
	// - Failed means that the eviction request is no longer being processed
	//   by any eviction responder. This can happen if the request is canceled or if no responder
	//   managed to evict the target (e.g. terminate or delete a pod).
	// - Evicted means that the target has been evicted (e.g. a pod has been terminated or deleted).
	//
	// 	The maximum length of the conditions list is 500.
	// +optional
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	// +k8s:optional
	// +k8s:listType=map
	// +k8s:listMapKey=type
	// +k8s:maxItems=500
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,1,rep,name=conditions"`

	// observedGeneration is EvictionRequest's .metadata.generation observed by the evictionrequest-controller.
	// The observed generation value cannot be negative and can only be incremented.
	// The minimum value is 1.
	// This field is managed by evictionrequest-controller.
	// +optional
	// +k8s:optional
	// +k8s:minimum=1
	ObservedGeneration *int64 `json:"observedGeneration,omitempty" protobuf:"varint,2,opt,name=observedGeneration"`

	// targetResponders reference responders that should eventually respond to this eviction
	// request to help with the graceful eviction of a target. These responders are selected
	// sequentially, in the order in which they appear in the list by setting the Active state to
	// the TargetResponder .state field. The maximum number of active responders allowed is 1.
	// Eventually each responder can end up in an Interrupted, Canceled or, Complete state.
	// Responders should observe these states in order to navigate their lifecycle.
	//
	// If the target is a pod, the field is populated from Pod's .spec.evictionResponders. Default
	// responders may be added to the list according to the target.
	//
	// Default responders:
	// - imperative-eviction.k8s.io/evictor responder is appended to the end of the list if the
	//   target is a pod. It will call the /evict API endpoint. This call may not succeed due to
	//   PodDisruptionBudgets, which may block the pod termination. It will update the responder
	//   message and try again with a backoff.
	//
	// The maximum length of the responders list is 17.
	// The length and keys of the list cannot change once set.
	// This field is managed by evictionrequest-controller.
	// +optional
	// +patchMergeKey=name
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=name
	// +k8s:optional
	// +k8s:listType=map
	// +k8s:listMapKey=name
	// +k8s:maxItems=17
	TargetResponders []TargetResponder `json:"targetResponders,omitempty" patchStrategy:"merge" patchMergeKey:"name" protobuf:"bytes,3,rep,name=targetResponders"`

	// responders represents the eviction process status of each declared responder.
	//
	// The responder list should be the same length and have the same .name fields as
	// .status.targetResponders. Only responders with .name that have Active state in
	// .targetResponders[].state should be updated and can be mutated. First initialization
	// of the list is allowed.
	//
	// Each ResponderStatus is initialized by evictionrequest-controller and then managed by
	// the designated responder.
	// +optional
	// +patchMergeKey=name
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=name
	// +k8s:optional
	// +k8s:listType=map
	// +k8s:listMapKey=name
	// +k8s:maxItems=17
	Responders []ResponderStatus `json:"responders,omitempty" patchStrategy:"merge" patchMergeKey:"name" protobuf:"bytes,6,rep,name=responders"`
}

type EvictionRequestConditionType string

// These are built-in conditions of an eviction request.
const (
	// EvictionRequestConditionFailed means that the eviction request is no longer being processed
	// by any eviction responder. This can happen if the request is canceled or if no responder
	// managed to evict the target (e.g. terminate or delete a pod).
	EvictionRequestConditionFailed EvictionRequestConditionType = "Failed"

	// EvictionRequestConditionEvicted means that the target has been evicted (e.g. a pod has been
	// terminated or deleted).
	EvictionRequestConditionEvicted EvictionRequestConditionType = "Evicted"
)

type EvictionRequestConditionReason string

// These are built-in condition reasons of an eviction request.
const (
	// EvictionRequestConditionReasonAwaitingEviction means that this EvictionRequest works as expected and the target
	// is scheduled for an eviction.
	// This reason is set for the Failed and Evicted condition.
	EvictionRequestConditionReasonAwaitingEviction EvictionRequestConditionReason = "AwaitingEviction"
	// EvictionRequestConditionReasonEvictionRequestInvalid means that the EvictionRequest is not accepted because the
	// initial configuration is not valid.
	// This reason is set for the Failed condition.
	EvictionRequestConditionReasonEvictionRequestInvalid EvictionRequestConditionReason = "EvictionRequestInvalid"
	// EvictionRequestConditionReasonCanceledDueToNoRequesters means that the EvictionRequest is canceled because it has
	// no requesters with an eviction intent.
	// This reason is set for the Failed condition.
	EvictionRequestConditionReasonCanceledDueToNoRequesters EvictionRequestConditionReason = "CanceledDueToNoRequesters"
	// EvictionRequestConditionReasonSucceeded means that the EvictionRequest has successfully evicted the target.
	// This reason is set for the Failed condition.
	EvictionRequestConditionReasonSucceeded EvictionRequestConditionReason = "Succeeded"
	// EvictionRequestConditionReasonNoFurtherResponder means that the EvictionRequest responders failed to evict
	// the target and that no further responder is available.
	// This reason is set for the Failed condition.
	EvictionRequestConditionReasonNoFurtherResponder EvictionRequestConditionReason = "NoFurtherResponder"
	// EvictionRequestConditionReasonPodDeleted means that the target pod has been deleted.
	// This reason is set for the Evicted condition.
	EvictionRequestConditionReasonPodDeleted EvictionRequestConditionReason = "PodDeleted"
	// EvictionRequestConditionReasonPodTerminal means that the target pod has reached a terminal state.
	// This reason is set for the Evicted condition.
	EvictionRequestConditionReasonPodTerminal EvictionRequestConditionReason = "PodTerminal"
	// EvictionRequestConditionReasonEvictionFailed means that the eviction of the target was unsuccessful.
	// This reason is set for the Evicted condition.
	EvictionRequestConditionReasonEvictionFailed EvictionRequestConditionReason = "EvictionFailed"
)

// TargetResponder allows you to specify the responder reacting to the EvictionRequest.
// Responders should observe and communicate through the EvictionRequest API (see .state) to help
// with the graceful eviction of a target (e.g. termination of a pod).
// +structType=atomic
type TargetResponder struct {
	// name allows you to identify the responder reacting to the EvictionRequest.
	//
	// It must be a valid domain-prefixed path (such as "acme.io/foo").
	// This field must be unique for each responder.
	// This field is required.
	// +required
	// +k8s:required
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`

	// state specifies a state that is assigned by the evictionrequest-controller. Responders should observe
	// this state in order to navigate their lifecycle.
	// - Inactive means that the responder should not yet process this eviction request.
	// - Active means that the responder is either running or expected to start soon.
	//   Also, startTime has been set in the ResponderStatus by the evictionrequest-controller.
	//
	//   An active responder should currently interact with the eviction process by updating
	//   .status.responders, where .name is the active responder name. ResponderStatus fields
	//   should be periodically updated to indicate the progress or completion of the eviction process.
	//   If .status.responders[].heartbeatTime field is not updated within 20 minutes, the eviction
	//   request is passed over to the next responder. Only one responder can be active at a time.
	// - Interrupted means that the responder has failed to start or failed to update
	//   heartbeatTime in ResponderStatus in a timely manner.
	// - Canceled means that the responder has been canceled. In other words, there is no
	//   Eviction intent in .spec.requesters.
	// - Completed means that the responder has successfully completed and set completionTime
	//   in ResponderStatus.
	//
	// Please refer to the ResponderStatus in .status.responders for more details on each responder.
	// +required
	// +k8s:required
	State ResponderStateType `json:"state" protobuf:"bytes,2,opt,name=state,casttype=ResponderStateType"`
}

// ResponderStateType specifies a state that is assigned by the evictionrequest-controller.
// +k8s:enum
type ResponderStateType string

const (
	// ResponderStateInactive means that the responder should not yet process this eviction request.
	ResponderStateInactive ResponderStateType = "Inactive"

	// ResponderStateActive means that the responder is either running or expected to start soon.
	// Also, startTime has been set in the ResponderStatus by the evictionrequest-controller.
	//
	// An active responder should currently interact with the eviction process by updating
	// .status.responders, where .name is the active responder name. ResponderStatus fields
	// should be periodically updated to indicate the progress or completion of the eviction process.
	// If .status.responders[].heartbeatTime field is not updated within 20 minutes, the eviction
	// request is passed over to the next responder. Only one responder can be active at a time.
	ResponderStateActive ResponderStateType = "Active"

	// ResponderStateInterrupted means that the responder has failed to start or failed to update
	// heartbeatTime in ResponderStatus in a timely manner.
	ResponderStateInterrupted ResponderStateType = "Interrupted"

	// ResponderStateCanceled means that the responder has been canceled. In other words, there is no
	// Eviction intent in .spec.requesters.
	ResponderStateCanceled ResponderStateType = "Canceled"

	// ResponderStateCompleted means that the responder has successfully completed and set completionTime
	// in ResponderStatus.
	ResponderStateCompleted ResponderStateType = "Completed"
)

// ResponderStatus represents the last observed status of the eviction process of the responder.
// It should be only updated by the designated responder whose name is .name field.
// +structType=granular
type ResponderStatus struct {
	// name allows you to identify the responder reacting to the EvictionRequest.
	//
	// It must be a valid domain-prefixed path (such as "acme.io/foo").
	// This field is initialized by Kubernetes and must be unique for each responder.
	// This field is required.
	// +required
	// +k8s:required
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`

	// startTime tracks the time at which this responder was designated as active and should start
	// processing the eviction request.
	// It should reflect the present time when set.
	// This field is initialized by Kubernetes when this responder becomes active.
	// This field becomes immutable once set.
	// +optional
	// +k8s:optional
	// +k8s:update=NoModify
	// +k8s:update=NoUnset
	StartTime *metav1.Time `json:"startTime,omitempty" protobuf:"bytes,2,opt,name=startTime"`

	// heartbeatTime is the last time at which the eviction process was reported to be in progress
	// by the responder.
	// It should reflect the present time when set.
	// Responders should avoid heartbeats more frequent than 20 seconds to avoid overloading the
	// control-plane.
	// +optional
	// +k8s:optional
	HeartbeatTime *metav1.Time `json:"heartbeatTime,omitempty" protobuf:"bytes,3,opt,name=heartbeatTime"`

	// expectedCompletionTime is the time at which the eviction process step is expected to end for the
	// responder.
	// The time cannot be set to the past.
	// May be omitted if no estimate can be made.
	// +optional
	// +k8s:optional
	ExpectedCompletionTime *metav1.Time `json:"expectedCompletionTime,omitempty" protobuf:"bytes,4,opt,name=expectedCompletionTime"`

	// completionTime tracks the time at which the Responder stopped processing the eviction request.
	// Completion means that the responders has either fully or partially completed the
	// eviction process, which may have resulted in target eviction (e.g. pod termination).
	// It should reflect the present time when set.
	// This field becomes immutable once set.
	// +optional
	// +k8s:optional
	// +k8s:update=NoModify
	// +k8s:update=NoUnset
	CompletionTime *metav1.Time `json:"completionTime,omitempty" protobuf:"bytes,5,opt,name=completionTime"`

	// message provides human-readable details about the state of the responder and the eviction
	// process.
	// Maximum length is 4000 characters.
	// +optional
	// +k8s:optional
	// +k8s:maxLength=4000
	Message string `json:"message,omitempty" protobuf:"bytes,6,opt,name=message"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.36

// EvictionRequestList contains a list of EvictionRequests resources.
type EvictionRequestList struct {
	metav1.TypeMeta `json:",inline"`
	// metadata is the standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is the list of EvictionRequests.
	Items []EvictionRequest `json:"items" protobuf:"bytes,2,rep,name=items"`
}
