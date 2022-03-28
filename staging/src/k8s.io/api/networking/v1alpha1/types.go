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

// CIDRConfig defines the CIDR and Mask size per IP Family(IPv4/IPv6).
type CIDRConfig struct {
	// Nodes may only have 1 range from each family.
	// An IP block in CIDR notation ("10.0.0.0/8", "fd12:3456:789a:1::/64").
	CIDR string `json:"cidr" protobuf:"bytes,1,name=cidr"`

	// PerNodeMaskSize is the mask size for node cidr.
	// IPv4/IPv6 Netmask size (e.g. 25 -> "/25" or 112 -> "/112") to allocate to a node.
	// Users would have to ensure that the kubelet doesn't try to schedule
	// more pods than are supported by the node's netmask (i.e. the kubelet's
	// --max-pods flag).
	PerNodeMaskSize int32 `json:"perNodeMaskSize" protobuf:"bytes,2,name=perNodeMaskSize"`
}

// ClusterCIDRConfigSpec defines the desired state of ClusterCIDRConfig.
type ClusterCIDRConfigSpec struct {
	// NodeSelector defines which nodes the config is applicable to.
	// An empty or nil NodeSelector functions as a default that applies to all nodes.
	// +optional
	NodeSelector *v1.NodeSelector `json:"nodeSelector,omitempty" protobuf:"bytes,1,opt,name=nodeSelector"`

	// IPv4 defines the IPv4 CIDR and the PerNodeMaskSize.
	// Atleast one of the IPv4 or IPv6 must be provided.
	// +optional
	IPv4 *CIDRConfig `json:"ipv4,omitempty" protobuf:"bytes,2,opt,name=ipv4"`

	// IPv6 defines the IPv6 CIDR and the PerNodeMaskSize.
	// Atleast one of the IPv4 or IPv6 must be provided.
	// +optional
	IPv6 *CIDRConfig `json:"ipv6,omitempty" protobuf:"bytes,3,opt,name=ipv6"`
}

// ClusterCIDRConfigStatus defines the observed state of ClusterCIDRConfig.
type ClusterCIDRConfigStatus struct {
	// Conditions contain details for the last reported state of ClusterCIDRConfig.
	//
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,1,rep,name=conditions"`
}
