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

// EvictionRequest defines an eviction request
type EvictionRequest struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Spec defines the eviction request specification.
	// This field is required.
	Spec EvictionRequestSpec

	// Status represents the most recently observed status of the eviction request.
	// Populated by the current interceptor and eviction request controller.
	// +optional
	Status EvictionRequestStatus
}

// EvictionRequestSpec is a specification of an EvictionRequest.
type EvictionRequestSpec struct {
	// PodRef references a pod that is subject to eviction/termination.
	// This field is required and immutable.
	PodRef LocalPodReference

	// Interceptors reference interceptors that respond to this eviction request.
	// This field does not need to be set and is resolved when the EvictionRequest object is created
	// on admission.
	// If no interceptors are specified, the pod in PodRef is evicted using the Eviction API.
	// The maximum length of the interceptors list is 100.
	// This field is immutable.
	// +optional
	Interceptors []Interceptor

	// HeartbeatDeadlineSeconds is a maximum amount of time that an interceptor should take to
	// periodically report on an eviction progress by updating the .status.heartbeatTime.
	// If the .status.heartbeatTime is not updated within the duration of
	// HeartbeatDeadlineSeconds, the eviction request is passed over to the next interceptor with the
	// highest priority. If there is none, the pod is evicted using the Eviction API.
	//
	// The minimum value is 600 (10m) and the maximum value is 21600 (6h).
	// The default value is 1800 (30m).
	// This field is required and immutable.
	HeartbeatDeadlineSeconds *int32
}

// LocalPodReference contains enough information to locate the referenced pod inside the same namespace.
type LocalPodReference struct {
	// Name of the pod.
	// This field is required.
	Name string
	// UID of the pod.
	// This field is required.
	UID string
}

// Interceptor information that allows you to identify the interceptor responding to this eviction
// request. Pods can be annotated with:
// interceptor.evictionrequest.coordination.k8s.io/priority_${INTERCEPTOR_CLASS}: ${PRIORITY}/${ROLE}
// interceptor.evictionrequest.coordination.k8s.io/priority_replicaset.apps.k8s.io: "10000/controller"
// annotations that can be parsed into the Interceptor struct when the EvictionRequest object is
// created on admission.
type Interceptor struct {
	// InterceptorClass must be RFC-1123 DNS subdomain identifying the interceptor (e.g.
	// foo.example.com).
	// This field is required.
	InterceptorClass string

	// Priority for this InterceptorClass. Higher priorities are selected first by the eviction
	// request controller. The interceptor that is the managing controller should set the value of
	// this field to 10000 to allow both for preemption or fallback registration by other
	// interceptors.
	//
	// Priorities 9900-10100 are reserved for interceptors with a class that has the same parent
	// domain as the controller interceptor. Duplicate priorities are not allowed in this interval.
	//
	// The number of interceptor annotations is limited to 30 in the 9900-10100 interval and to 70
	// outside of this interval.
	// The minimum value is 0 and the maximum value is 100000.
	// This field is required.
	Priority int32

	// Role of the interceptor. The "controller" value is reserved for the managing controller of
	// the pod. The role can send additional signal to other interceptors if they should preempt
	// this interceptor or not.
	// +optional
	Role *string
}

// EvictionRequestStatus represents the last observed status of the eviction request.
type EvictionRequestStatus struct {
	// Conditions can be used by interceptors to share additional information about the eviction
	// request.
	// +optional
	Conditions []metav1.Condition

	// Message is a human readable message indicating details about the eviction request.
	// This may be an empty string.
	Message string

	// Interceptors of the ActiveInterceptorClass can adopt this eviction request by updating the
	// HeartbeatTime or orphan/complete it by setting ActiveInterceptorCompleted to true.
	// +optional
	ActiveInterceptorClass *string

	// ActiveInterceptorCompleted should be set to true when the interceptor of the
	// ActiveInterceptorClass has fully or partially completed (may result in pod termination).
	// This field can also be set to true if no interceptor is available.
	// If this field is true, there is no additional interceptor available, and the evicted pod is
	// still running, it will be evicted using the Eviction API.
	// +optional
	ActiveInterceptorCompleted bool

	// ExpectedInterceptorFinishTime is the time at which the eviction process step is expected to
	// end for the current interceptor and its class.
	// May be empty if no estimate can be made.
	// +optional
	ExpectedInterceptorFinishTime *metav1.Time

	// HeartbeatTime is the time at which the eviction process was reported to be in progress by
	// the interceptor.
	// Cannot be set to the future time (after taking time skew of up to 10 seconds into account).
	// +optional
	HeartbeatTime *metav1.Time

	// EvictionRequestCancellationPolicy should be set to Forbid by the interceptor if it is not possible
	// to cancel (delete) the eviction request.
	// When this value is Forbid, DELETE requests of this EvictionRequest object will not be accepted
	// while the pod exists.
	// This field is not reset by the eviction request controller when selecting an interceptor.
	// Changes to this field should always be reconciled by the active interceptor.
	//
	// Valid policies are Allow and Forbid.
	// The default value is Allow.
	//
	// Allow policy allows cancellation of this eviction request.
	// The EvictionRequest can be deleted before the Pod is fully terminated.
	//
	// Forbid policy forbids cancellation of this eviction request.
	// The EvictionRequest can't be deleted until the Pod is fully terminated.
	//
	// This field is required.
	EvictionRequestCancellationPolicy EvictionRequestCancellationPolicy

	// The number of unsuccessful attempts to evict the referenced pod via the API-initiated eviction,
	// e.g. due to a PodDisruptionBudget.
	// This is set by the eviction controller after all the interceptors have completed.
	// The minimum value is 1, and subsequent updates can only increase it.
	// +optional
	FailedAPIEvictionCounter *int32
}

type EvictionRequestCancellationPolicy string

const (
	// Allow policy allows cancellation of this eviction request.
	// The EvictionRequest can be deleted before the Pod is fully terminated.
	Allow EvictionRequestCancellationPolicy = "Allow"
	// Forbid policy forbids cancellation of this eviction request.
	// The EvictionRequest can't be deleted until the Pod is fully terminated.
	Forbid EvictionRequestCancellationPolicy = "Forbid"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// EvictionRequestList is a collection of EvictionRequests.
type EvictionRequestList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta
	// Items is a list of EvictionRequests
	Items []EvictionRequest
}
