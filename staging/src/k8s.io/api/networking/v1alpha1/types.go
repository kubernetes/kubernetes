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
// +k8s:prerelease-lifecycle-gen:introduced=1.24

// ClusterCIDRConfig is the Schema for the clustercidrconfigs API.
type ClusterCIDRConfig struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec is the desired state of the ClusterCIDRConfig.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Spec ClusterCIDRConfigSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// Status is the current state of the ClusterCIDRConfig.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Status ClusterCIDRConfigStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// ClusterCIDRConfigSpec defines the desired state of ClusterCIDRConfig.
type ClusterCIDRConfigSpec struct {
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
	// This field is immutable.
	// +optional
	PerNodeHostBits int32 `json:"perNodeHostBits" protobuf:"varint,2,opt,name=perNodeHostBits"`

	// IPv4CIDR defines an IPv4 IP block in CIDR notation(e.g. "10.0.0.0/8").
	// This field is immutable.
	// +optional
	IPv4CIDR string `json:"ipv4CIDR" protobuf:"bytes,3,opt,name=ipv4CIDR"`

	// IPv6CIDR defines an IPv6 IP block in CIDR notation(e.g. "fd12:3456:789a:1::/64").
	// This field is immutable.
	// +optional
	IPv6CIDR string `json:"ipv6CIDR" protobuf:"bytes,4,opt,name=ipv6CIDR"`
}

// ClusterCIDRConfigStatus defines the observed state of ClusterCIDRConfig.
type ClusterCIDRConfigStatus struct {
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.24

// ClusterCIDRConfigList contains a list of ClusterCIDRConfig.
type ClusterCIDRConfigList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of ClusterCIDRConfigs.
	Items []ClusterCIDRConfig `json:"items" protobuf:"bytes,2,rep,name=items"`
}
