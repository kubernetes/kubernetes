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

type CoordinatedStrategy string

// CoordinatedLeaseStrategy defines the strategy for picking the leader for coordinated leader election.
const (
	// OldestCompatibilityVersion picks the oldest compatibility version
	// by first picking the lowest binary version, and then selecting the
	// lowest compatibility version if binary versions match.
	// If there are multiple with the same version, then the
	// leader with the lowest lexicographical comparison result based on the name
	// will be selected.
	OldestCompatibilityVersion CoordinatedStrategy = "OldestCompatibilityVersion"
	// NoCoordination opts out of coordinated leader election
	// and allows leader election to run without a centralized manager.
	NoCoordination CoordinatedStrategy = "NoCoordination"
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
	// Strategy indicates the strategy for picking the leader for coordinated leader election
	// +optional
	Strategy *CoordinatedStrategy
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

// LeaseCandidate defines a candidate for a lease object.
// Candidates are created such that coordinated leader election will pick the best leader from the list of candidates.
type LeaseCandidate struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta
	Spec LeaseCandidateSpec
}

// LeaseCandidateSpec is a specification of a Lease.
type LeaseCandidateSpec struct {
	// BinaryVersion is the binary version
	BinaryVersion string
	// CompatibilityVersion is the compatibility version
	CompatibilityVersion string
	// TargetLease is the name of the lease that the candidate can lead
	TargetLease string

	// leaseDurationSeconds is a duration that candidates for a lease need
	// to wait to force acquire it. This is measure against time of last
	// observed renewTime.
	// +optional
	LeaseDurationSeconds *int32
	// renewTime is a time when the current holder of a lease has last
	// updated the lease.
	// +optional
	RenewTime *metav1.MicroTime
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
