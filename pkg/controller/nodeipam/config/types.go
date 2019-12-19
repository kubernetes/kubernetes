/*
Copyright 2019 The Kubernetes Authors.

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

package config

// NodeIPAMControllerConfiguration contains elements describing NodeIPAMController.
type NodeIPAMControllerConfiguration struct {
	// serviceCIDR is CIDR Range for Services in cluster.
	ServiceCIDR string
	// secondaryServiceCIDR is CIDR Range for Services in cluster. This is used in dual stack clusters. SecondaryServiceCIDR must be of different IP family than ServiceCIDR
	SecondaryServiceCIDR string
	// NodeCIDRMaskSize is the mask size for node cidr in single-stack cluster.
	// This can be used only with single stack clusters and is incompatible with dual stack clusters.
	NodeCIDRMaskSize int32
	// NodeCIDRMaskSizeIPv4 is the mask size for IPv4 node cidr in dual-stack cluster.
	// This can be used only with dual stack clusters and is incompatible with single stack clusters.
	NodeCIDRMaskSizeIPv4 int32
	// NodeCIDRMaskSizeIPv6 is the mask size for IPv6 node cidr in dual-stack cluster.
	// This can be used only with dual stack clusters and is incompatible with single stack clusters.
	NodeCIDRMaskSizeIPv6 int32
}
