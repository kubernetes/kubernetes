/*
Copyright 2025 The Kubernetes Authors.

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

// +k8s:deepcopy-gen=package

// This is a test package to validate deepcopy generation for value copyable types.
package copyable

import (
	"net/netip"
)

// NetworkConfig tests various netip types in different contexts
type NetworkConfig struct {
	// Direct value types
	ServerAddr netip.Addr
	CIDR       netip.Prefix
	Endpoint   netip.AddrPort

	// Pointer types
	GatewayAddr *netip.Addr
	Subnet      *netip.Prefix

	// Slice types
	DNSServers []netip.Addr
	AllowedIPs []netip.Prefix

	// Map types
	NamedAddresses map[string]netip.Addr
	RoutingTable   map[netip.Prefix]netip.Addr
}
