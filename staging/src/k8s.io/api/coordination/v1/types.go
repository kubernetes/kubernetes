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

package v1

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

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.14

// Lease defines a lease concept.
type Lease struct {
	metav1.TypeMeta `json:",inline"`
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec contains the specification of the Lease.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Spec LeaseSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
}

// LeaseSpec is a specification of a Lease.
type LeaseSpec struct {
	// holderIdentity contains the identity of the holder of a current lease.
	// If Coordinated Leader Election is used, the holder identity must be
	// equal to the elected LeaseCandidate.metadata.name field.
	// +optional
	HolderIdentity *string `json:"holderIdentity,omitempty" protobuf:"bytes,1,opt,name=holderIdentity"`
	// leaseDurationSeconds is a duration that candidates for a lease need
	// to wait to force acquire it. This is measured against the time of last
	// observed renewTime.
	// +optional
	LeaseDurationSeconds *int32 `json:"leaseDurationSeconds,omitempty" protobuf:"varint,2,opt,name=leaseDurationSeconds"`
	// acquireTime is a time when the current lease was acquired.
	// +optional
	AcquireTime *metav1.MicroTime `json:"acquireTime,omitempty" protobuf:"bytes,3,opt,name=acquireTime"`
	// renewTime is a time when the current holder of a lease has last
	// updated the lease.
	// +optional
	RenewTime *metav1.MicroTime `json:"renewTime,omitempty" protobuf:"bytes,4,opt,name=renewTime"`
	// leaseTransitions is the number of transitions of a lease between
	// holders.
	// +optional
	LeaseTransitions *int32 `json:"leaseTransitions,omitempty" protobuf:"varint,5,opt,name=leaseTransitions"`
	// Strategy indicates the strategy for picking the leader for coordinated leader election.
	// If the field is not specified, there is no active coordination for this lease.
	// (Alpha) Using this field requires the CoordinatedLeaderElection feature gate to be enabled.
	// +featureGate=CoordinatedLeaderElection
	// +optional
	Strategy *CoordinatedLeaseStrategy `json:"strategy,omitempty" protobuf:"bytes,6,opt,name=strategy"`
	// PreferredHolder signals to a lease holder that the lease has a
	// more optimal holder and should be given up.
	// This field can only be set if Strategy is also set.
	// +featureGate=CoordinatedLeaderElection
	// +optional
	PreferredHolder *string `json:"preferredHolder,omitempty" protobuf:"bytes,7,opt,name=preferredHolder"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.19

// LeaseList is a list of Lease objects.
type LeaseList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is a list of schema objects.
	Items []Lease `json:"items" protobuf:"bytes,2,rep,name=items"`
}
