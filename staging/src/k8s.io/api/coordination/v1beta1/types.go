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

package v1beta1

import (
	v1 "k8s.io/api/coordination/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.12
// +k8s:prerelease-lifecycle-gen:deprecated=1.19
// +k8s:prerelease-lifecycle-gen:replacement=coordination.k8s.io,v1,Lease

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
	// to wait to force acquire it. This is measure against time of last
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
	// Strategy indicates the strategy for picking the leader for coordinated leader election
	// (Alpha) Using this field requires the CoordinatedLeaderElection feature gate to be enabled.
	// +featureGate=CoordinatedLeaderElection
	// +optional
	Strategy *v1.CoordinatedLeaseStrategy `json:"strategy,omitempty" protobuf:"bytes,6,opt,name=strategy"`
	// PreferredHolder signals to a lease holder that the lease has a
	// more optimal holder and should be given up.
	// +featureGate=CoordinatedLeaderElection
	// +optional
	PreferredHolder *string `json:"preferredHolder,omitempty" protobuf:"bytes,7,opt,name=preferredHolder"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.12
// +k8s:prerelease-lifecycle-gen:deprecated=1.19
// +k8s:prerelease-lifecycle-gen:replacement=coordination.k8s.io,v1,LeaseList

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

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.33

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
	// The limits on this field are the same as on Lease.name. Multiple lease candidates
	// may reference the same Lease.name.
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
// +k8s:prerelease-lifecycle-gen:introduced=1.33

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
