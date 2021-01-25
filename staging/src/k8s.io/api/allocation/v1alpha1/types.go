/*
Copyright 2021 The Kubernetes Authors.

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
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.21

// IPRange defines a range of IPs using CIDR format (192.168.0.0/24 or 2001:db2::0/64).
type IPRange struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// +optional
	Spec IPRangeSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
	// +optional
	Status IPRangeStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// IPRangeSpec describe how the IPRange's specification looks like.
type IPRangeSpec struct {
	// Range of IPs in CIDR format (192.168.0.0/24 or 2001:db2::0/64).
	Range string `json:"range,omitempty" protobuf:"bytes,1,name=range"`
	// Primary indicates if this is the primary allocator to be used by the
	// apiserver to allocate IP addresses.
	// NOTE this can simplify the Service strategy logic so we don't have to infer
	// the primary allocator, it also may allow to switch between primary families in
	// a cluster, but this looks like a loooong shot.
	// +optional
	Primary bool `json:"primary,omitempty" protobuf:"bytes,2,opt,name=primary"`
}

// IPRangeStatus defines the observed state of IPRange.
type IPRangeStatus struct {
	// Free represent the number of IP addresses that are not allocated in the Range.
	// +optional
	Free int64 `json:"free,omitempty" protobuf:"bytes,1,opt,name=free"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.21

// IPRangeList contains a list of IPRange objects.
type IPRangeList struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items           []IPRange `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// QUESTION if the IPAddress is namespaced and it's owned by the service associated
// the garbage collector can delete it once the Service is deleted.
// If affirmative, we can get rid of the Allocator.Interface and the Release() step
// that wills simplify the Strategy of services and fix some known problems when
// deleting services

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.21

// IPAddress represents an IP used by Kubernetes associated to an IPRange.
// The name of the object is the IP address decimal number, because colons
// are not allowed and IPv6 addresses have different text representations.
// xref: https://tools.ietf.org/html/rfc4291
type IPAddress struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// +optional
	Spec IPAddressSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
	// +optional
	Status IPAddressStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// IPRangeRef contains information that points to the IPRange being used so we can validate it
type IPRangeRef struct {
	// APIGroup is the group for the resource being referenced
	APIGroup string `json:"apiGroup" protobuf:"bytes,1,opt,name=apiGroup"`
	// Kind is the type of resource being referenced
	Kind string `json:"kind" protobuf:"bytes,2,opt,name=kind"`
	// Name is the name of resource being referenced
	Name string `json:"name" protobuf:"bytes,3,opt,name=name"`
}

// IPAddressSpec describe the attributes in an IP Address,
type IPAddressSpec struct {
	// Address is the text representation of the IP Address.
	Address string `json:"address" protobuf:"bytes,1,name=address"`
	// IPRangeRef references the IPRange associated to this IP Address.
	// All IP addresses has to be associated to one IPRange allocator.
	IPRangeRef IPRangeRef `json:"ipRangeRef,omitempty" protobuf:"bytes,2,name=ipRangeRef"`
}

// IPAddressStatus defines the observed state of IPAddress.
type IPAddressStatus struct {
	State IPAddressState `json:"state,omitempty" protobuf:"bytes,1,name=state"`
}

// IPAddressState defines the state of the IP address
type IPAddressState string

// These are the valid statuses of IPAddresses.
const (
	// IPPending means the IP has been allocated by the system but the object associated
	// (typically Services ClusterIPs) has not been persisted yet.
	IPPending IPAddressState = "Pending"
	// IPAllocated means the IP has been persisted with the object associated.
	IPAllocated IPAddressState = "Allocated"
	// IPFree means that IP has not been allocated neither persisted.
	IPFree IPAddressState = "Free"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.21

// IPAddressList contains a list of IPAddress.
type IPAddressList struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items           []IPAddress `json:"items" protobuf:"bytes,2,rep,name=items"`
}
