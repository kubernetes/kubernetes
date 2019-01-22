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

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Lease defines a lease concept.
type Lease struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Specification of the Lease.
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
	// observed RenewTime.
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
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// LeaseList is a list of Lease objects.
type LeaseList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	// Items is a list of schema objects.
	Items []Lease
}
