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

package allocation

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// IPRange defines a range of IPs using CIDR format (192.168.0.0/24 or 2001:db2::0/64).
type IPRange struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   IPRangeSpec   `json:"spec,omitempty"`
	Status IPRangeStatus `json:"status,omitempty"`
}

// IPRangeSpec describe how the IPRange's specification looks like.
type IPRangeSpec struct {
	// Range of IPs in CIDR format (192.168.0.0/24 or 2001:db2::0/64).
	Range string `json:"range"`
	// Primary indicates if this is the primary allocator to be used by the
	// apiserver to allocate IP addresses.
	// NOTE this can simplify the Service strategy logic so we don't have to infer
	// the primary allocator, it also may allow to switch between primary families in
	// a cluster, but this looks like a loooong shot.
	// +optional
	Primary bool `json:"primary,omitempty"`
}

// IPRangeStatus defines the observed state of IPRange.
type IPRangeStatus struct {
	// Free represent the number of IP addresses that are not allocated in the Range.
	// +optional
	Free int64 `json:"free,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// IPRangeList contains a list of IPRange objects.
type IPRangeList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []IPRange `json:"items"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// IPAddress represents an IP used by Kubernetes associated to an IPRange.
// The name of the object is the IP address decimal number, because colons
// are not allowed and IPv6 addresses have different text representations.
// xref: https://tools.ietf.org/html/rfc4291
type IPAddress struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   IPAddressSpec   `json:"spec,omitempty"`
	Status IPAddressStatus `json:"status,omitempty"`
}

// IPRangeRef contains information that points to the IPRange being used
type IPRangeRef struct {
	// APIGroup is the group for the resource being referenced
	APIGroup string
	// Kind is the type of resource being referenced
	Kind string
	// Name is the name of resource being referenced
	Name string
}

// IPAddressSpec describe the attributes in an IP Address,
type IPAddressSpec struct {
	// Address is the text representation of the IP Address.
	Address string `json:"address"`
	// IPRangeRef references the IPRange associated to this IP Address.
	IPRangeRef IPRangeRef `json:"ipRangeRef,omitempty"`
}

// IPAddressStatus defines the observed state of IPAddress.
type IPAddressStatus struct {
	State IPAddressState `json:"state,omitempty"`
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

// IPAddressList contains a list of IPAddress.
type IPAddressList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []IPAddress `json:"items"`
}
