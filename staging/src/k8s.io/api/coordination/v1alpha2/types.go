/*
Copyright 2024 The Kubernetes Authors.

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

package v1alpha2

import (
	v1 "k8s.io/api/coordination/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.32

// LeaseCandidate defines a candidate for a Lease object.
// Candidates are created such that coordinated leader election will pick the best leader from the list of candidates.
type LeaseCandidate struct {
	metav1.TypeMeta `json:",inline"`
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec contains the specification of the Lease.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Spec LeaseCandidateSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
}

// LeaseCandidateSpec is a specification of a Lease.
type LeaseCandidateSpec struct {
	// LeaseName is the name of the lease for which this candidate is contending.
	// This field is immutable.
	// +required
	LeaseName string `json:"leaseName" protobuf:"bytes,1,name=leaseName"`
	// PingTime is the last time that the server has requested the LeaseCandidate
	// to renew. It is only done during leader election to check if any
	// LeaseCandidates have become ineligible. When PingTime is updated, the
	// LeaseCandidate will respond by updating RenewTime.
	// +optional
	PingTime *metav1.MicroTime `json:"pingTime,omitempty" protobuf:"bytes,2,opt,name=pingTime"`
	// RenewTime is the time that the LeaseCandidate was last updated.
	// Any time a Lease needs to do leader election, the PingTime field
	// is updated to signal to the LeaseCandidate that they should update
	// the RenewTime.
	// Old LeaseCandidate objects are also garbage collected if it has been hours
	// since the last renew. The PingTime field is updated regularly to prevent
	// garbage collection for still active LeaseCandidates.
	// +optional
	RenewTime *metav1.MicroTime `json:"renewTime,omitempty" protobuf:"bytes,3,opt,name=renewTime"`
	// BinaryVersion is the binary version. It must be in a semver format without leading `v`.
	// This field is required.
	// +required
	BinaryVersion string `json:"binaryVersion" protobuf:"bytes,4,name=binaryVersion"`
	// EmulationVersion is the emulation version. It must be in a semver format without leading `v`.
	// EmulationVersion must be less than or equal to BinaryVersion.
	// This field is required when strategy is "OldestEmulationVersion"
	// +optional
	EmulationVersion string `json:"emulationVersion,omitempty" protobuf:"bytes,5,opt,name=emulationVersion"`
	// Strategy is the strategy that coordinated leader election will use for picking the leader.
	// If multiple candidates for the same Lease return different strategies, the strategy provided
	// by the candidate with the latest BinaryVersion will be used. If there is still conflict,
	// this is a user error and coordinated leader election will not operate the Lease until resolved.
	// +required
	Strategy v1.CoordinatedLeaseStrategy `json:"strategy,omitempty" protobuf:"bytes,6,opt,name=strategy"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.32

// LeaseCandidateList is a list of Lease objects.
type LeaseCandidateList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is a list of schema objects.
	Items []LeaseCandidate `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.35

// EvictionRequest defines an eviction request
type EvictionRequest struct {
	metav1.TypeMeta `json:",inline"`

	// Object's metadata.
	// .metadata.generateName is not supported.
	// .metadata.name should match the .metadata.uid of the pod being evicted.
	// The labels of the eviction request object will be merged with pod's .metadata.labels. The
	// labels of the pod have a preference.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec defines the eviction request specification.
	// https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// This field is required.
	// +required
	Spec EvictionRequestSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// Status represents the most recently observed status of the eviction request.
	// Populated by the current interceptor and eviction request controller.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Status EvictionRequestStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// EvictionRequestSpec is a specification of an EvictionRequest.
type EvictionRequestSpec struct {
	// PodRef references a pod that is subject to eviction/termination.
	// This field is required and immutable.
	// +required
	PodRef LocalPodReference `json:"podRef" protobuf:"bytes,1,opt,name=podRef"`

	// Interceptors reference interceptors that respond to this eviction request.
	// This field does not need to be set and is resolved when the EvictionRequest object is created
	// on admission.
	// If no interceptors are specified, the pod in PodRef is evicted using the Eviction API.
	// The maximum length of the interceptors list is 100.
	// This field is immutable.
	// +optional
	// +patchMergeKey=interceptorClass
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=interceptorClass
	Interceptors []Interceptor `json:"interceptors,omitempty"  patchStrategy:"merge" patchMergeKey:"interceptorClass" protobuf:"bytes,2,rep,name=interceptors"`

	// HeartbeatDeadlineSeconds is a maximum amount of time that an interceptor should take to
	// periodically report on an eviction progress by updating the .status.heartbeatTime.
	// If the .status.heartbeatTime is not updated within the duration of
	// HeartbeatDeadlineSeconds, the eviction request is passed over to the next interceptor with the
	// highest priority. If there is none, the pod is evicted using the Eviction API.
	//
	// The minimum value is 600 (10m) and the maximum value is 21600 (6h).
	// The default value is 1800 (30m).
	// This field is required and immutable.
	// +required
	HeartbeatDeadlineSeconds *int32 `json:"heartbeatDeadlineSeconds" protobuf:"varint,3,opt,name=heartbeatDeadlineSeconds"`
}

// LocalPodReference contains enough information to locate the referenced pod inside the same namespace.
type LocalPodReference struct {
	// Name of the pod.
	// This field is required.
	// +required
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`
	// UID of the pod.
	// This field is required.
	// +required
	UID string `json:"uid" protobuf:"bytes,2,opt,name=uid"`
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
	// +required
	InterceptorClass string `json:"interceptorClass" protobuf:"bytes,1,opt,name=interceptorClass"`

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
	// +required
	Priority int32 `json:"priority" protobuf:"varint,2,opt,name=priority"`

	// Role of the interceptor. The "controller" value is reserved for the managing controller of
	// the pod. The role can send additional signal to other interceptors if they should preempt
	// this interceptor or not.
	// +optional
	Role *string `json:"role,omitempty" protobuf:"bytes,3,opt,name=role"`
}

// EvictionRequestStatus represents the last observed status of the eviction request.
type EvictionRequestStatus struct {
	// Conditions can be used by interceptors to share additional information about the eviction
	// request.
	// +optional
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,8,rep,name=conditions"`

	// Message is a human readable message indicating details about the eviction request.
	// This may be an empty string.
	// +required
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxLength=32768
	Message string `json:"message" protobuf:"bytes,7,opt,name=message"`

	// Interceptors of the ActiveInterceptorClass can adopt this eviction request by updating the
	// HeartbeatTime or orphan/complete it by setting ActiveInterceptorCompleted to true.
	// +optional
	ActiveInterceptorClass *string `json:"activeInterceptorClass,omitempty" protobuf:"bytes,1,opt,name=activeInterceptorClass"`

	// ActiveInterceptorCompleted should be set to true when the interceptor of the
	// ActiveInterceptorClass has fully or partially completed (may result in pod termination).
	// This field can also be set to true if no interceptor is available.
	// If this field is true, there is no additional interceptor available, and the evicted pod is
	// still running, it will be evicted using the Eviction API.
	// +optional
	ActiveInterceptorCompleted bool `json:"activeInterceptorCompleted,omitempty" protobuf:"varint,2,opt,name=activeInterceptorCompleted"`

	// ExpectedInterceptorFinishTime is the time at which the eviction process step is expected to
	// end for the current interceptor and its class.
	// May be empty if no estimate can be made.
	// +optional
	ExpectedInterceptorFinishTime *metav1.Time `json:"expectedInterceptorFinishTime,omitempty" protobuf:"bytes,3,opt,name=expectedInterceptorFinishTime"`

	// HeartbeatTime is the time at which the eviction process was reported to be in progress by
	// the interceptor.
	// Cannot be set to the future time (after taking time skew of up to 10 seconds into account).
	// +optional
	HeartbeatTime *metav1.Time `json:"heartbeatTime,omitempty" protobuf:"bytes,4,opt,name=heartbeatTime"`

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
	// +required
	EvictionRequestCancellationPolicy EvictionRequestCancellationPolicy `json:"evictionRequestCancellationPolicy" protobuf:"varint,5,opt,name=evictionRequestCancellationPolicy"`

	// The number of unsuccessful attempts to evict the referenced pod via the API-initiated eviction,
	// e.g. due to a PodDisruptionBudget.
	// This is set by the eviction controller after all the interceptors have completed.
	// The minimum value is 1, and subsequent updates can only increase it.
	// +optional
	FailedAPIEvictionCounter *int32 `json:"failedAPIEvictionCounter,omitempty" protobuf:"varint,6,opt,name=failedAPIEvictionCounter"`
}

// +enum
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
// +k8s:prerelease-lifecycle-gen:introduced=1.35

// EvictionRequestList is a collection of EvictionRequests.
type EvictionRequestList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// Items is a list of EvictionRequests
	Items []EvictionRequest `json:"items" protobuf:"bytes,2,rep,name=items"`
}
