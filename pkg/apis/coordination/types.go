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
	"k8s.io/kubernetes/pkg/apis/core"
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

// KubernetesEvictionInterceptorName is the name identifying a core in-tree interceptor.
type KubernetesEvictionInterceptorName string

const (
	// EvictionInterceptorImperativeEviction is a default interceptor that will evict pods using the imperative
	// Eviction API (/evict endpoint) with a backoff.
	EvictionInterceptorImperativeEviction KubernetesEvictionInterceptorName = "imperative-eviction.k8s.io"
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
// `.spec.requesters` field should be set and kept updated to preserve the eviction request.
//
// If the target is a pod, the .status.targetInterceptors is populated from Pod's
// .spec.evictionInterceptors.
//
// Interceptors should observe and communicate through the .status to help with the eviction
// of the target when they see their name present in .status.activeInterceptors. InterceptorStatus
// struct should then be periodically updated to indicate the progress or completion of the eviction
// process by each interceptor in .status.intercerptors. If .status.interceptors[].heartbeatTime is
// not updated within 20 minutes, the eviction request is passed over to the next interceptor.
//
// If there are no other interceptor and the target is a pod, the last default
// imperative-eviction.k8s.io interceptor will evict the pod using the imperative Eviction API
// (/evict endpoint).
type EvictionRequest struct {
	metav1.TypeMeta

	// Object's metadata.
	// .metadata.name should match the .metadata.uid of the pod being evicted.
	// .metadata.generateName is not supported.
	// The labels of the eviction request object are synchronized with .metadata.labels of the
	// eviction request's target. The labels of the target have a preference.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta

	// Spec defines the eviction request specification.
	// https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +required
	Spec EvictionRequestSpec

	// Status represents the most recently observed status of the eviction request.
	// Populated by interceptors and eviction request controller.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Status EvictionRequestStatus
}

// EvictionRequestSpec is a specification of an EvictionRequest.
type EvictionRequestSpec struct {
	// Target contains a reference to an object (e.g. a pod) that should be evicted.
	// Target UID must be the same as the EvictionRequest's .metadata.name.
	// This field is immutable.
	// +required
	Target EvictionTarget

	// Requesters allow you to identify entities, that requested the eviction of the target.
	// At least one requester is required when creating an eviction request.
	// A requester is also required for the eviction request to be processed.
	// Empty list indicates that the eviction request should be canceled.
	//
	// Requester controllers are expected to reconcile this field and not overwrite any changes made
	// by other controllers (e.g. via Server-Side Apply) in order to prevent conflicts and manage
	// ownership.
	//
	// The maximum length of the requesters list is 100.
	// +optional
	// +patchMergeKey=name
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=name
	Requesters []Requester
}

// EvictionTarget contains a reference to an object that should be evicted.
// +union
type EvictionTarget struct {
	// Pod references a pod that is subject to eviction/termination.
	// Pods that are part of the workload (.spec.workloadRef is set) are not supported.
	// +optional
	// +oneOf=TargetSelection
	Pod *LocalTargetReference
}

// LocalTargetReference contains enough information to locate the referenced target inside the same namespace.
type LocalTargetReference struct {
	// Name of the target.
	// This field is required.
	// +required
	Name string
	// UID of the target.
	// This field is required.
	// +required
	UID string
}

// Requester allows you to identify the entity, that requested the eviction of the target.
// +structType=atomic
type Requester struct {
	// Name must be a fully qualified domain name of at most 253 characters in length, consisting
	// only of lowercase alphanumeric characters, periods and hyphens (e.g. foo.example.com).
	// This field must be unique for each requester.
	// This field is required.
	// +required
	Name string
}

// EvictionRequestStatus represents the last observed status of the eviction request.
type EvictionRequestStatus struct {
	// EvictionRequest's .metadata.generation observed by the eviction request controller.
	// The observed generation value cannot be negative and can only be incremented.
	// This field is managed by Kubernetes.
	// +optional
	ObservedGeneration int64

	// TargetInterceptors reference interceptors that should eventually respond to this eviction
	// request to help with the graceful eviction of a target. These interceptors are selected
	// sequentially, in the order in which they appear in the list and are added to the
	// .status.activeInterceptors in a rolling fashion.
	//
	// If the target is a pod, the field is populated from Pod's .spec.evictionInterceptors. Default
	// interceptors may be added to the list according to the target.
	//
	// Default interceptors:
	// - imperative-eviction.k8s.io interceptor is appended to the end of the list if the target is
	//   a pod. It will call the /evict API endpoint. This call may not succeed due to
	//   PodDisruptionBudgets, which may block the pod termination. It will update the interceptor
	//   message and try again with a backoff.
	//
	// The maximum length of the interceptors list is 16.
	// This field is immutable once set.
	// This field is managed by Kubernetes.
	// +optional
	// +patchMergeKey=name
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=name
	TargetInterceptors []core.EvictionInterceptor

	// ActiveInterceptors store a list of interceptors that should currently interact with the
	// eviction process by updating .status.interceptors[], where .name is the active interceptor
	// name. InterceptorStatus fields should be periodically updated to indicate the progress or
	// completion of the eviction process. If .status.interceptors[].heartbeatTime field is not
	// updated within 20 minutes, the eviction request is passed over to the next interceptor.
	//
	// The maximum allowed number of active interceptors is 1. An active interceptor is removed from
	// this list when .status.interceptors[].completionTime is set.
	// This field is managed by Kubernetes.
	// +optional
	// +listType=atomic
	ActiveInterceptors []string

	// ProcessedInterceptors store a list of interceptors that have previously been selected
	// and added to ActiveInterceptors. These interceptors may have reached completion, been
	// canceled, failed to start or failed to update .interceptors[].heartbeatTime` in a timely
	// manner.
	// Please refer to the InterceptorStatus for each interceptor for more details.
	// This field is managed by Kubernetes.
	// +optional
	// +listType=atomic
	ProcessedInterceptors []string

	// Interceptors represents the eviction process status of each declared interceptor. Only
	// ActiveInterceptors should update the interceptor statuses.
	//
	// The interceptor list should be the same length and have the same .name fields as
	// .status.targetInterceptors. Only interceptors with .name that are included in
	// .status.activeInterceptors can be mutated. First initialization of the list is allowed.
	//
	// Each InterceptorStatus is managed by the designated interceptor.
	// +optional
	// +patchMergeKey=name
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=name
	Interceptors []InterceptorStatus

	// Conditions contain information about the eviction request.
	//
	// Eviction request specific conditions are: Evicted or Canceled (managed by Kubernetes),
	//
	// - Canceled means that the eviction request is no longer being processed by any eviction
	//   interceptor because the eviction request has been canceled.
	// - Evicted means that the target has been evicted (e.g. a pod has been terminated or deleted).
	// +optional
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition
}

type EvictionRequestConditionType string

// These are built-in conditions of an eviction request.
const (
	// EvictionRequestConditionCanceled means that the eviction request is no longer being processed
	// by any eviction interceptor because the eviction request has been canceled.
	EvictionRequestConditionCanceled EvictionRequestConditionType = "Canceled"

	// EvictionRequestConditionEvicted means that the target has been evicted (e.g. a pod has been
	// terminated or deleted).
	EvictionRequestConditionEvicted EvictionRequestConditionType = "Evicted"
)

type EvictionRequestConditionReason string

// These are built-in condition reasons of an eviction request.
const (
	// EvictionRequestConditionReasonNoRequesters means that the EvictionRequest has no requesters and is set for the Canceled condition.
	EvictionRequestConditionReasonNoRequesters EvictionRequestConditionReason = "NoRequesters"
	// EvictionRequestConditionReasonValidationFailed means that the EvictionRequest is not valid and is set for the Canceled condition.
	EvictionRequestConditionReasonValidationFailed EvictionRequestConditionReason = "ValidationFailed"
	// EvictionRequestConditionReasonPodDeleted means that the target pod has been deleted and is set for the Evicted condition.
	EvictionRequestConditionReasonPodDeleted EvictionRequestConditionReason = "PodDeleted"
	// EvictionRequestConditionReasonPodTerminal means that the target pod has reached a terminal state and is set for the Evicted condition.
	EvictionRequestConditionReasonPodTerminal EvictionRequestConditionReason = "PodTerminal"
)

// InterceptorStatus represents the last observed status of the eviction process of the interceptor.
// It should be only updated by the designated interceptor whose name is .name field.
// +structType=atomic
type InterceptorStatus struct {
	// Name must be a fully qualified domain name of at most 253 characters in length, consisting
	// only of lowercase alphanumeric characters, periods and hyphens (e.g. bar.example.com).
	// This field is initialized by Kubernetes and must be unique for each interceptor.
	// This field is required and immutable.
	// +required
	Name string

	// StartTime tracks the time at which this interceptor was designated as active and should start
	// processing the eviction request.
	// It should reflect the present time when set.
	// This field is initialized by Kubernetes when this interceptor becomes active.
	// This field becomes immutable once set.
	// +optional
	StartTime *metav1.Time

	// HeartbeatTime is the last time at which the eviction process was reported to be in progress
	// by the interceptor.
	// It should reflect the present time when set.
	// There must be at least 60 second increments during subsequent updates.
	// +optional
	HeartbeatTime *metav1.Time

	// ExpectedCompletionTime is the time at which the eviction process step is expected to end for the
	// interceptor.
	// The time cannot be set to the past.
	// May be empty if no estimate can be made.
	// +optional
	ExpectedCompletionTime *metav1.Time

	// CompletionTime tracks the time at which the Interceptor stopped processing the eviction request.
	// Completion means that the interceptors has either fully or partially completed the
	// eviction process, which may have resulted in target eviction (e.g. pod termination).
	// It should reflect the present time when set.
	// This field becomes immutable once set.
	// +optional
	CompletionTime *metav1.Time

	// Message provides human-readable details about the state of the interceptor and the eviction
	// process.
	// Maximum length is 4000 characters. The string is truncated if it exceeds this limit.
	// +optional
	Message string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// EvictionRequestList contains a list of EvictionRequests resources.
type EvictionRequestList struct {
	metav1.TypeMeta
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta

	// Items is the list of EvictionRequests.
	Items []EvictionRequest
}
