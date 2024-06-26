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
	// BinaryVersion is the binary version
	BinaryVersion string `json:"binaryVersion,omitempty" protobuf:"bytes,5,opt,name=binaryVersion"`
	// CompatibilityVersion is the compatibility version
	CompatibilityVersion string `json:"compatibilityVersion,omitempty" protobuf:"bytes,6,opt,name=compatiblityVersion"`
	// TargetLease is the name of the lease that the candidate can lead
	TargetLease string `json:"targetLease,omitempty" protobuf:"bytes,7,opt,name=targetLease"`

	// leaseDurationSeconds is a duration that candidates for a lease need
	// to wait to force acquire it. This is measure against time of last
	// observed renewTime.
	// +optional
	LeaseDurationSeconds *int32 `json:"leaseDurationSeconds,omitempty" protobuf:"varint,2,opt,name=leaseDurationSeconds"`
	// renewTime is the time that the LeaseCandidate was last updated.
	// Unlike Lease objects, candidates are not refreshed frequently.
	// Any time a Lease needs to do leader election, an annotation is sent
	// to all the candidates to renew their candidacy and update the
	// renewTime here.
	// +optional
	RenewTime *metav1.MicroTime `json:"renewTime,omitempty" protobuf:"bytes,4,opt,name=renewTime"`
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
