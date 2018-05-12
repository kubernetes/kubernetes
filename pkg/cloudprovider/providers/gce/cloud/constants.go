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

package cloud

import (
	"strings"
)

// NetworkTier represents the Network Service Tier used by a resource
type NetworkTier string

// LbScheme represents the possible types of load balancers
type LbScheme string

const (
	NetworkTierStandard NetworkTier = "Standard"
	NetworkTierPremium  NetworkTier = "Premium"
	NetworkTierDefault  NetworkTier = NetworkTierPremium

	SchemeExternal LbScheme = "EXTERNAL"
	SchemeInternal LbScheme = "INTERNAL"
)

// ToGCEValue converts NetworkTier to a string that we can populate the
// NetworkTier field of GCE objects, including ForwardingRules and Addresses.
func (n NetworkTier) ToGCEValue() string {
	return strings.ToUpper(string(n))
}

// NetworkTierGCEValueToType converts the value of the NetworkTier field of a
// GCE object to the NetworkTier type.
func NetworkTierGCEValueToType(s string) NetworkTier {
	switch s {
	case NetworkTierStandard.ToGCEValue():
		return NetworkTierStandard
	case NetworkTierPremium.ToGCEValue():
		return NetworkTierPremium
	default:
		return NetworkTier(s)
	}
}
