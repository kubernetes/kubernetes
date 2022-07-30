/*
Copyright 2022 The Kubernetes Authors.

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
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.25

// ClusterCIDR is the Schema for the ClusterCIDR API.
// This resource is consumed by the MultiCIDRRangeAllocator to allocate pod CIDRs
// to nodes. When there are multiple ClusterCIDR resources in the cluster, the
// list of all applicable ClusterCIDR resources is collected. A ClusterCIDR is
// applicable if its NodeSelector matches the Node being allocated, and if it has
// free CIDRs to allocate.
// In case of multiple matching ClusterCIDR resources, MultiCIDRRangeAllocator
// attempts to break ties sequentially using the following rules:
// 1. Pick the ClusterCIDR whose NodeSelector matches the most labels/fields on the Node.
// For example, Pick {'node.kubernetes.io/instance-type': 'medium', 'rack': 'rack1'}
// before {'node.kubernetes.io/instance-type': 'medium'}
// 2. Pick the ClusterCIDR with the fewest allocatable Pod CIDRs.
// For example, {IPv4: "10.0.0.0/16", PerNodeHostBits: "16"} (1 possible Pod CIDR)
// is picked before {IPv4: "192.168.0.0/20", PerNodeHostBits: "10"} (4 possible Pod CIDRs)
// 3. Pick the ClusterCIDR whose PerNodeHostBits is the fewest IPs.
// For example, {PerNodeHostBits: 5} (32 IPs) picked before {PerNodeHostBits: 7} (128 IPs).
// 4. Pick the ClusterCIDR having label with lower alphanumeric value.
// For example, Pick {'node.kubernetes.io/instance-type': 'low'}
// before {'node.kubernetes.io/instance-type': 'medium'}
// 5. Pick the ClusterCIDR having a smaller IP address value.
// For example, Pick {IPv4: "10.0.0.0/16"} before {IPv4: "10.11.0.0/16"}
type ClusterCIDR struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec is the desired state of the ClusterCIDR.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Spec ClusterCIDRSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
}

// ClusterCIDRSpec defines the desired state of ClusterCIDR.
type ClusterCIDRSpec struct {
	// NodeSelector defines which nodes the config is applicable to.
	// An empty or nil NodeSelector functions as a default that applies to all nodes.
	// This field is immutable.
	// +optional
	NodeSelector *v1.NodeSelector `json:"nodeSelector,omitempty" protobuf:"bytes,1,opt,name=nodeSelector"`

	// PerNodeHostBits defines the number of host bits to be configured per node.
	// A subnet mask determines how much of the address is used for network bits
	// and host bits. For example and IPv4 address of 192.168.0.0/24, splits the
	// address into 24 bits for the network portion and 8 bits for the host portion.
	// For a /24 mask for IPv4 or a /120 for IPv6, configure PerNodeHostBits=8
	// Minimum value for PerNodeHostBits is 4.
	// This field is immutable.
	// +required
	PerNodeHostBits int32 `json:"perNodeHostBits" protobuf:"varint,2,opt,name=perNodeHostBits"`

	// IPv4 defines an IPv4 IP block in CIDR notation(e.g. "10.0.0.0/8").
	// This field is immutable.
	// +optional
	IPv4 string `json:"ipv4" protobuf:"bytes,3,opt,name=ipv4"`

	// IPv6 defines an IPv6 IP block in CIDR notation(e.g. "fd12:3456:789a:1::/64").
	// This field is immutable.
	// +optional
	IPv6 string `json:"ipv6" protobuf:"bytes,4,opt,name=ipv6"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.25

// ClusterCIDRList contains a list of ClusterCIDR.
type ClusterCIDRList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of ClusterCIDRs.
	Items []ClusterCIDR `json:"items" protobuf:"bytes,2,rep,name=items"`
}
