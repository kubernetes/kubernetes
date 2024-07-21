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

package v1alpha1

import (
	v1 "k8s.io/api/coordination/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.31

// LeaseCandidate defines a candidate for a lease object.
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

// LeaseSpec is a specification of a Lease.
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
	// Old LeaseCandidate objects are also garbage collected if it has been hours since the last renew.
	// +optional
	RenewTime *metav1.MicroTime `json:"renewTime,omitempty" protobuf:"bytes,3,opt,name=renewTime"`
	// BinaryVersion is the binary version. It must be in a semver format without leading `v`.
	// This field is required when Strategy is "OldestEmulationVersion"
	// +optional
	BinaryVersion string `json:"binaryVersion,omitempty" protobuf:"bytes,4,opt,name=binaryVersion"`
	// EmulationVersion is the emulation version. It must be in a semver format without leading `v`.
	// EmulationVersion must be less than or equal to BinaryVersion.
	// This field is required when Strategy is "OldestEmulationVersion"
	// +optional
	EmulationVersion string `json:"emulationVersion,omitempty" protobuf:"bytes,5,opt,name=emulationVersion"`
	// PreferredStrategies indicates the list of strategies for picking the leader for coordinated leader election.
	// The list is ordered, and the first strategy supersedes all other strategies. The list is used by coordinated
	// leader election to make a decision about the final election strategy. This follows as
	// - If all clients have strategy X as the first element in this list, strategy X will be used.
	// - If a candidate has strategy [X] and another candidate has strategy [Y, X], Y supersedes X and strategy Y
	//   will be used
	// - If a candidate has strategy [X, Y] and another candidate has strategy [Y, X], this is a user error and leader
	//   election will not operate the Lease until resolved.
	// (Alpha) Using this field requires the CoordinatedLeaderElection feature gate to be enabled.
	// +featureGate=CoordinatedLeaderElection
	// +listType=atomic
	// +required
	PreferredStrategies []v1.CoordinatedLeaseStrategy `json:"preferredStrategies,omitempty" protobuf:"bytes,6,opt,name=preferredStrategies"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.31

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
