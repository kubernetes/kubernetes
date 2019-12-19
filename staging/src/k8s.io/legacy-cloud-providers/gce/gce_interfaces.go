// +build !providerless

/*
Copyright 2017 The Kubernetes Authors.

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

package gce

import (
	computealpha "google.golang.org/api/compute/v0.alpha"
	computebeta "google.golang.org/api/compute/v0.beta"
	compute "google.golang.org/api/compute/v1"
)

// These interfaces are added for testability.

// CloudAddressService is an interface for managing addresses
type CloudAddressService interface {
	ReserveRegionAddress(address *compute.Address, region string) error
	GetRegionAddress(name string, region string) (*compute.Address, error)
	GetRegionAddressByIP(region, ipAddress string) (*compute.Address, error)
	DeleteRegionAddress(name, region string) error
	// TODO: Mock Global endpoints

	// Alpha API.
	GetAlphaRegionAddress(name, region string) (*computealpha.Address, error)
	ReserveAlphaRegionAddress(addr *computealpha.Address, region string) error

	// Beta API
	ReserveBetaRegionAddress(address *computebeta.Address, region string) error
	GetBetaRegionAddress(name string, region string) (*computebeta.Address, error)
	GetBetaRegionAddressByIP(region, ipAddress string) (*computebeta.Address, error)

	// TODO(#51665): Remove this once the Network Tiers becomes Alpha in GCP.
	getNetworkTierFromAddress(name, region string) (string, error)
}

// CloudForwardingRuleService is an interface for managing forwarding rules.
// TODO: Expand the interface to include more methods.
type CloudForwardingRuleService interface {
	GetRegionForwardingRule(name, region string) (*compute.ForwardingRule, error)
	CreateRegionForwardingRule(rule *compute.ForwardingRule, region string) error
	DeleteRegionForwardingRule(name, region string) error

	// Alpha API.
	GetAlphaRegionForwardingRule(name, region string) (*computealpha.ForwardingRule, error)
	CreateAlphaRegionForwardingRule(rule *computealpha.ForwardingRule, region string) error

	// Needed for the Alpha "Network Tiers" feature.
	getNetworkTierFromForwardingRule(name, region string) (string, error)
}
