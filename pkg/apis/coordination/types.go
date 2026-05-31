/*
Copyright 2018 The Kubernetes Authors.

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

package coordination

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apimachinerytypes "k8s.io/apimachinery/pkg/types"
)

type CoordinatedLeaseStrategy string

// CoordinatedLeaseStrategy defines the strategy for picking the leader for coordinated leader election.
const (
	// OldestEmulationVersion picks the oldest LeaseCandidate, where "oldest" is defined as follows
	// 1) Select the candidate(s) with the lowest emulation version
	// 2) If multiple candidates have the same emulation version, select the candidate(s) with the lowest binary version. (Note that binary version must be greater or equal to emulation version)
	// 3) If multiple candidates have the same binary version, select the candidate with the oldest creationTimestamp.
	// If a candidate does not specify the emulationVersion and binaryVersion fields, it will not be considered a candidate for the lease.
	OldestEmulationVersion CoordinatedLeaseStrategy = "OldestEmulationVersion"
)

// KubernetesEvictionResponderName is the name identifying a core in-tree responder.
type KubernetesEvictionResponderName string

const (
	// EvictionResponderImperativeEviction is a default responder that will evict pods using the imperative
	// Eviction API (/evict endpoint) with a backoff.
	EvictionResponderImperativeEviction KubernetesEvictionResponderName = "imperative-eviction.k8s.io/evictor"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Lease defines a lease concept.
type Lease struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// spec contains the specification of the Lease.
	// +optional
	Spec LeaseSpec
}

// LeaseSpec is a specification of a Lease.
type LeaseSpec struct {
	// holderIdentity contains the identity of the holder of a current lease.
	// If Coordinated Leader Election is used, the holder identity must be
	// equal to the elected LeaseCandidate.metadata.name field.
	// +optional
	HolderIdentity *string
	// leaseDurationSeconds is a duration that candidates for a lease need
	// to wait to force acquire it. This is measure against time of last
	// observed renewTime.
	// +optional
	LeaseDurationSeconds *int32
	// acquireTime is a time when the current lease was acquired.
	// +optional
	AcquireTime *metav1.MicroTime
	// renewTime is a time when the current holder of a lease has last
	// updated the lease.
	// +optional
	RenewTime *metav1.MicroTime
	// leaseTransitions is the number of transitions of a lease between
	// holders.
	// +optional
	LeaseTransitions *int32
	// Strategy indicates the strategy for picking the leader for coordinated leader election.
	// If the field is not specified, there is no active coordination for this lease.
	// (Alpha) Using this field requires the CoordinatedLeaderElection feature gate to be enabled.
	// +featureGate=CoordinatedLeaderElection
	// +optional
	Strategy *CoordinatedLeaseStrategy
	// PreferredHolder signals to a lease holder that the lease has a
	// more optimal holder and should be given up.
	// This field can only be set if Strategy is also set.
	// +featureGate=CoordinatedLeaderElection
	// +optional
	PreferredHolder *string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// LeaseList is a list of Lease objects.
type LeaseList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	// items is a list of schema objects.
	Items []Lease
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// LeaseCandidate defines a candidate for a Lease object.
// Candidates are created such that coordinated leader election will pick the best leader from the list of candidates.
type LeaseCandidate struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta
	Spec LeaseCandidateSpec
}

// LeaseCandidateSpec is a specification of a Lease.
type LeaseCandidateSpec struct {
	// LeaseName is the name of the lease for which this candidate is contending.
	// The limits on this field are the same as on Lease.name. Multiple lease candidates
	// may reference the same Lease.name.
	// This field is immutable.
	// +required
	LeaseName string
	// PingTime is the last time that the server has requested the LeaseCandidate
	// to renew. It is only done during leader election to check if any
	// LeaseCandidates have become ineligible. When PingTime is updated, the
	// LeaseCandidate will respond by updating RenewTime.
	// +optional
	PingTime *metav1.MicroTime
	// RenewTime is the time that the LeaseCandidate was last updated. Any time
	// a Lease needs to do leader election, the PingTime field is updated to
	// signal to the LeaseCandidate that they should update the RenewTime. The
	// PingTime field is also updated regularly and LeaseCandidates must update
	// RenewTime to prevent garbage collection for still active LeaseCandidates.
	// Old LeaseCandidate objects are periodically garbage collected.
	// +optional
	RenewTime *metav1.MicroTime
	// BinaryVersion is the binary version. It must be in a semver format without leading `v`.
	// This field is required.
	// +required
	BinaryVersion string
	// EmulationVersion is the emulation version. It must be in a semver format without leading `v`.
	// EmulationVersion must be less than or equal to BinaryVersion.
	// This field is required when strategy is "OldestEmulationVersion"
	// +optional
	EmulationVersion string
	// Strategy is the strategy that coordinated leader election will use for picking the leader.
	// If multiple candidates for the same Lease return different strategies, the strategy provided
	// by the candidate with the latest BinaryVersion will be used. If there is still conflict,
	// this is a user error and coordinated leader election will not operate the Lease until resolved.
	// +listType=atomic
	// +required
	Strategy CoordinatedLeaseStrategy
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// LeaseCandidateList is a list of LeaseCandidate objects.
type LeaseCandidateList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	// items is a list of schema objects.
	Items []LeaseCandidate
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// EvictionRequest defines a request that should ideally result in a graceful eviction of a
// .spec.target (e.g. termination of a pod).
//
// The evictionrequest-controller observes intents of all EvictionRequests and transforms them into
// Evictions.
//   - .spec.requesterName is set as a label on the Eviction for easier lookup.
//   - Each target can have a set of responders assigned to it. Eviction objects are observed by
//     these responders, who implement the eviction logic and update the Eviction's status with
//     progress.
//
// There is many-to-many relationship between EvictionRequests and Evictions.
//
// If all requesters withdraw their eviction intent for a common target, the eviction will be
// canceled. Deleting an EvictionRequest also counts as a withdrawal.
// Once all EvictionRequest of a target are removed, the corresponding Evictions are eventually
// garbage collected.
type EvictionRequest struct {
	metav1.TypeMeta

	// metadata is the standard object metadata; More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata.
	// +optional
	metav1.ObjectMeta

	// spec defines the eviction request specification.
	// https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +required
	Spec EvictionRequestSpec

	// status represents the most recently observed status of the eviction request.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Status EvictionRequestStatus
}

// EvictionRequestSpec is a specification of an EvictionRequest.
type EvictionRequestSpec struct {
	// target contains a reference to an object (e.g. a pod) that should be evicted.
	// This field is required and immutable.
	// +required
	Target EvictionRequestTarget

	// requesterName allows you to identify the entity, that requested the eviction of the target.
	//
	// It must be a valid domain-prefixed path (such as "acme.io/foo").
	// Domain names *.k8s.io and *.kubernetes.io are reserved.
	// This field is required and immutable.
	// +required
	RequesterName string

	// intent specifies the action that should be taken for the specified target.
	//
	// - Eviction means that the requester is interested in the eviction of the target.
	// - Withdrawn means that the requester is no longer interested in the eviction of the target.
	//   If all requesters' intents are withdrawn for a common target, the eviction will be canceled.
	//   Cancellation consequences:
	//   - Inactive responders will never run.
	//   - Active responders are expected to cancel the eviction.
	//   - Completed or Interrupted responders should not take any action.
	// +required
	Intent EvictionRequestIntent
}

// EvictionRequestTarget contains a reference to an object that should be evicted.
type EvictionRequestTarget struct {
	// pod references a pod that is subject to eviction/termination.
	// Pods that are part of a PodGroup (.spec.schedulingGroup is set) are not supported.
	// +optional
	Pod *EvictionRequestPodReference
}

// EvictionRequestPodReference contains enough information to locate the referenced pod inside the
// same namespace.
type EvictionRequestPodReference struct {
	// name of the target.
	// This field is required.
	// +required
	Name string
	// uid of the target.
	// It can be found in .spec.metadata.uid of the target and is a lowercase UUID in 8-4-4-4-12 format.
	// This field is required.
	// +required
	UID apimachinerytypes.UID
}

// EvictionRequestIntent specifies a requester intent.
type EvictionRequestIntent string

// These are intents that can be set by each requester.
const (
	// EvictionRequestIntentEviction means that the requester is interested in the eviction of the target.
	EvictionRequestIntentEviction EvictionRequestIntent = "Eviction"

	// EvictionRequestIntentWithdrawn means that the requester is no longer interested in the eviction of the target.
	// If all requesters' intents are withdrawn for a common target, the eviction will be canceled.
	// Cancellation consequences:
	// - Inactive responders will never run.
	// - Active responders are expected to cancel the eviction.
	// - Completed or Interrupted responders should not take any action.
	EvictionRequestIntentWithdrawn EvictionRequestIntent = "Withdrawn"
)

// EvictionRequestStatus represents the last observed status of the eviction request.
type EvictionRequestStatus struct {
	// conditions contain information about the eviction request.
	//
	// EvictionRequest specific conditions are: Evicted or Failed (managed by evictionrequest-controller).
	// - Failed means that the eviction request is no longer being processed
	//   by any eviction responder. This can happen if the request is canceled or if no responder
	//   managed to evict the target (e.g. terminate or delete a pod).
	// - Evicted means that the target has been evicted (e.g. a pod has been terminated or deleted).
	//
	// These conditions can be reset if the eviction was unsuccessful and a new Eviction intent has
	// been submitted.
	//
	// The maximum length of the conditions list is 100.
	// +optional
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition

	// observedGeneration is EvictionRequest's .metadata.generation observed by the evictionrequest-controller.
	// The observed generation value cannot be negative and can only be incremented.
	// The minimum value is 1.
	// This field is managed by evictionrequest-controller.
	// +optional
	ObservedGeneration *int64
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// EvictionRequestList contains a list of EvictionRequests resources.
type EvictionRequestList struct {
	metav1.TypeMeta
	// metadata is the standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta

	// items is the list of EvictionRequests.
	Items []EvictionRequest
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Eviction initiates an eviction process, which should ideally result in a graceful eviction of a
// .spec.target (e.g. termination of a pod).
//
// The evictionrequest-controller observes intents of all EvictionRequests and transforms them into
// Evictions. It manages the Eviction lifecycle.
// Requesters are preserved in .status.requesters even after they have withdrawn their request.
// If all requesters withdraw their eviction intent for a common target, the eviction will be
// canceled. Once all EvictionRequest corresponding to this Eviction .spec.target have been
// removed, this Eviction object will eventually be garbage collected.
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
type Eviction struct {
	metav1.TypeMeta

	// metadata is the standard object metadata; More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata.
	// .metadaata.name set by the evictionrequest-controller is purely informative and subject to change.
	// .spec.target field should be used to identify the target precisesly.
	//
	// The requester and responder names will be used as label keys and added to the labels of the
	// eviction in one of the following formats:
	// 1. acme.io/foo: "requester"
	// 2. acme.io/foo: "responder"
	// 3. acme.io/foo: "requesterresponder"
	// +optional
	metav1.ObjectMeta

	// spec defines the eviction specification.
	// https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +required
	Spec EvictionSpec

	// status represents the most recently observed status of the eviction.
	// Populated by responders and evictionrequest-controller.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Status EvictionStatus
}

// EvictionSpec is a specification of an Eviction.
type EvictionSpec struct {
	// target contains a reference to an object (e.g. a pod) that should be evicted.
	// This field is required and immutable.
	// +required
	Target EvictionTarget
}

// EvictionTarget contains a reference to an object that should be evicted.
// +union
type EvictionTarget struct {
	// pod references a pod that is subject to eviction/termination.
	// Pods that are part of a PodGroup (.spec.schedulingGroup is set) are not supported.
	// +optional
	Pod *EvictionPodReference
}

// EvictionPodReference contains enough information to locate the referenced pod inside the same
// namespace.
type EvictionPodReference struct {
	// name of the target.
	// This field is required.
	// +required
	Name string
	// uid of the target.
	// It can be found in .spec.metadata.uid of the target and is a lowercase UUID in 8-4-4-4-12 format.
	// This field is required.
	// +required
	UID apimachinerytypes.UID
}

// EvictionStatus represents the last observed status of the eviction request.
type EvictionStatus struct {
	// conditions contain information about the eviction request.
	//
	// Eviction request specific conditions are: Evicted or Failed (managed by evictionrequest-controller).
	// - Failed means that the eviction request is no longer being processed
	//   by any eviction responder. This can happen if the request is canceled or if no responder
	//   managed to evict the target (e.g. terminate or delete a pod).
	// - Evicted means that the target has been evicted (e.g. a pod has been terminated or deleted).
	//
	// 	The maximum length of the conditions list is 100.
	// +optional
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition

	// observedGeneration is Eviction's .metadata.generation observed by the evictionrequest-controller.
	// The observed generation value cannot be negative and can only be incremented.
	// The minimum value is 1.
	// This field is managed by evictionrequest-controller.
	// +optional
	ObservedGeneration *int64

	// requesters allow you to identify the entities, that requested the eviction of the target.
	// If all the requesters withdraw their eviction intent, the eviction will be canceled.
	//
	// Once added, items cannot be removed.
	// +optional
	// +patchMergeKey=name
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=name
	Requesters []Requester

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
	TargetResponders []TargetResponder

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
	Responders []ResponderStatus
}

type EvictionConditionType string

// These are built-in conditions of an eviction request.
const (
	// EvictionConditionFailed means that the eviction request is no longer being processed
	// by any eviction responder. This can happen if the request is canceled or if no responder
	// managed to evict the target (e.g. terminate or delete a pod).
	EvictionConditionFailed EvictionConditionType = "Failed"

	// EvictionConditionEvicted means that the target has been evicted (e.g. a pod has been
	// terminated or deleted).
	EvictionConditionEvicted EvictionConditionType = "Evicted"
)

type EvictionConditionReason string

// These are built-in condition reasons of an eviction request.
const (
	// EvictionConditionReasonAwaitingEviction means that this Eviction works as expected and the target
	// is scheduled for an eviction.
	// This reason is set for the Failed and Evicted condition.
	EvictionConditionReasonAwaitingEviction EvictionConditionReason = "AwaitingEviction"
	// EvictionConditionReasonEvictionInvalid means that the Eviction is not accepted because the
	// initial configuration is not valid.
	// This reason is set for the Failed condition.
	EvictionConditionReasonEvictionInvalid EvictionConditionReason = "EvictionInvalid"
	// EvictionConditionReasonCanceledDueToNoRequesters means that the Eviction is canceled because there is no
	// EvictionRequest with the same target and Eviction intent in .spec.intent.
	// This reason is set for the Failed condition.
	EvictionConditionReasonCanceledDueToNoRequesters EvictionConditionReason = "CanceledDueToNoRequesters"
	// EvictionConditionReasonSucceeded means that the Eviction has successfully evicted the target.
	// This reason is set for the Failed condition.
	EvictionConditionReasonSucceeded EvictionConditionReason = "Succeeded"
	// EvictionConditionReasonNoFurtherResponder means that the Eviction responders failed to evict
	// the target and that no further responder is available.
	// This reason is set for the Failed condition.
	EvictionConditionReasonNoFurtherResponder EvictionConditionReason = "NoFurtherResponder"
	// EvictionConditionReasonPodDeleted means that the target pod has been deleted.
	// This reason is set for the Evicted condition.
	EvictionConditionReasonPodDeleted EvictionConditionReason = "PodDeleted"
	// EvictionConditionReasonPodTerminal means that the target pod has reached a terminal state.
	// This reason is set for the Evicted condition.
	EvictionConditionReasonPodTerminal EvictionConditionReason = "PodTerminal"
	// EvictionConditionReasonEvictionFailed means that the eviction of the target was unsuccessful.
	// This reason is set for the Evicted condition.
	EvictionConditionReasonEvictionFailed EvictionConditionReason = "EvictionFailed"
)

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
	Name string

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
	Intent RequesterIntent
}

// RequesterIntent specifies a requester intent.
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

// TargetResponder allows you to specify the responder reacting to the Eviction.
// Responders should observe and communicate through the Eviction API (see .state) to help
// with the graceful eviction of a target (e.g. termination of a pod).
// +structType=atomic
type TargetResponder struct {
	// name allows you to identify the responder reacting to the Eviction.
	//
	// It must be a valid domain-prefixed path (such as "acme.io/foo").
	// This field must be unique for each responder.
	// This field is required.
	// +required
	Name string

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
	// - Canceled means that the responder has been canceled. In other words, there	is no
	//   EvictionRequest with the same target and Eviction intent in .spec.intent.
	// - Completed means that the responder has successfully completed and set completionTime
	//   in ResponderStatus.
	//
	// Please refer to the ResponderStatus in .status.responders for more details on each responder.
	// +required
	State ResponderStateType
}

// ResponderStateType specifies a state that is assigned by the evictionrequest-controller.
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

	// ResponderStateCanceled means that the responder has been canceled. In other words, there
	// is no EvictionRequest with the same target and Eviction intent in .spec.intent.
	ResponderStateCanceled ResponderStateType = "Canceled"

	// ResponderStateCompleted means that the responder has successfully completed and set completionTime
	// in ResponderStatus.
	ResponderStateCompleted ResponderStateType = "Completed"
)

// ResponderStatus represents the last observed status of the eviction process of the responder.
// It should be only updated by the designated responder whose name is .name field.
// +structType=granular
type ResponderStatus struct {
	// name allows you to identify the responder reacting to the Eviction.
	//
	// It must be a valid domain-prefixed path (such as "acme.io/foo").
	// This field is initialized by Kubernetes and must be unique for each responder.
	// This field is required.
	// +required
	Name string

	// startTime tracks the time at which this responder was designated as active and should start
	// processing the eviction request.
	// It should reflect the present time when set.
	// This field is initialized by Kubernetes when this responder becomes active.
	// This field becomes immutable once set.
	// +optional
	StartTime *metav1.Time

	// heartbeatTime is the last time at which the eviction process was reported to be in progress
	// by the responder.
	// It should reflect the present time when set.
	// Responders should avoid heartbeats more frequent than 20 seconds to avoid overloading the
	// control-plane.
	// +optional
	HeartbeatTime *metav1.Time

	// expectedCompletionTime is the time at which the eviction process step is expected to end for the
	// responder.
	// The time cannot be set to the past.
	// May be omitted if no estimate can be made.
	// +optional
	ExpectedCompletionTime *metav1.Time

	// completionTime tracks the time at which the Responder stopped processing the eviction request.
	// Completion means that the responders has either fully or partially completed the
	// eviction process, which may have resulted in target eviction (e.g. pod termination).
	// It should reflect the present time when set.
	// This field becomes immutable once set.
	// +optional
	CompletionTime *metav1.Time

	// message provides human-readable details about the state of the responder and the eviction
	// process.
	// Maximum length is 4000 characters.
	// +optional
	Message string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// EvictionList contains a list of Eviction resources.
type EvictionList struct {
	metav1.TypeMeta
	// metadata is the standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta

	// items is the list of Evictions.
	Items []Eviction
}
