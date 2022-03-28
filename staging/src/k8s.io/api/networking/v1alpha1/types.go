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

	// IPv4 defines the IPv4 CIDR and the PerNodeMaskSize.
	// At least one of IPv4 or IPv6 must be provided. If both are
	// provided, the number of IPs allocated to each must be the same
	// (32 - ipv4.perNodeMaskSize).
	// This field is immutable.
	// +optional
	IPv4 *CIDRConfig `json:"ipv4,omitempty" protobuf:"bytes,2,opt,name=ipv4"`

	// IPv6 defines the IPv4 CIDR and the PerNodeMaskSize.
	// At least one of IPv4 or IPv6 must be provided. If both are
	// provided, the number of IPs allocated to each must be the same
	// (128 - ipv6.perNodeMaskSize).
	// This field is immutable.
	// +optional
	IPv6 *CIDRConfig `json:"ipv6,omitempty" protobuf:"bytes,3,opt,name=ipv6"`
}

// CIDRConfig defines the CIDR and Mask size per IP Family(IPv4/IPv6).
type CIDRConfig struct {
	// An IP block in CIDR notation ("10.0.0.0/8", "fd12:3456:789a:1::/64").
	CIDR string `json:"cidr" protobuf:"bytes,1,name=cidr"`

	// PerNodeMaskSize is the mask size for node cidr.
	// IPv4/IPv6 Netmask size (e.g. 25 -> "/25" or 112 -> "/112") to allocate to a node.
	PerNodeMaskSize int32 `json:"perNodeMaskSize" protobuf:"bytes,2,name=perNodeMaskSize"`
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
