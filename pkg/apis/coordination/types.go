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
