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
	"fmt"

	"github.com/golang/glog"

	computealpha "google.golang.org/api/compute/v0.alpha"
	computebeta "google.golang.org/api/compute/v0.beta"
	compute "google.golang.org/api/compute/v1"

	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/filter"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
)

func newAddressMetricContext(request, region string) *metricContext {
	return newAddressMetricContextWithVersion(request, region, computeV1Version)
}

func newAddressMetricContextWithVersion(request, region, version string) *metricContext {
	return newGenericMetricContext("address", request, region, unusedMetricLabel, version)
}

// ReserveGlobalAddress creates a global address.
// Caller is allocated a random IP if they do not specify an ipAddress. If an
// ipAddress is specified, it must belong to the current project, eg: an
// ephemeral IP associated with a global forwarding rule.
func (gce *GCECloud) ReserveGlobalAddress(addr *compute.Address) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newAddressMetricContext("reserve", "")
	return mc.Observe(gce.c.GlobalAddresses().Insert(ctx, meta.GlobalKey(addr.Name), addr))
}

// DeleteGlobalAddress deletes a global address by name.
func (gce *GCECloud) DeleteGlobalAddress(name string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newAddressMetricContext("delete", "")
	return mc.Observe(gce.c.GlobalAddresses().Delete(ctx, meta.GlobalKey(name)))
}

// GetGlobalAddress returns the global address by name.
func (gce *GCECloud) GetGlobalAddress(name string) (*compute.Address, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newAddressMetricContext("get", "")
	v, err := gce.c.GlobalAddresses().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// ReserveRegionAddress creates a region address
func (gce *GCECloud) ReserveRegionAddress(addr *compute.Address, region string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newAddressMetricContext("reserve", region)
	return mc.Observe(gce.c.Addresses().Insert(ctx, meta.RegionalKey(addr.Name, region), addr))
}

// ReserveAlphaRegionAddress creates an Alpha, regional address.
func (gce *GCECloud) ReserveAlphaRegionAddress(addr *computealpha.Address, region string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newAddressMetricContext("reserve", region)
	return mc.Observe(gce.c.AlphaAddresses().Insert(ctx, meta.RegionalKey(addr.Name, region), addr))
}

// ReserveBetaRegionAddress creates a beta region address
func (gce *GCECloud) ReserveBetaRegionAddress(addr *computebeta.Address, region string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newAddressMetricContext("reserve", region)
	return mc.Observe(gce.c.BetaAddresses().Insert(ctx, meta.RegionalKey(addr.Name, region), addr))
}

// DeleteRegionAddress deletes a region address by name.
func (gce *GCECloud) DeleteRegionAddress(name, region string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newAddressMetricContext("delete", region)
	return mc.Observe(gce.c.Addresses().Delete(ctx, meta.RegionalKey(name, region)))
}

// GetRegionAddress returns the region address by name
func (gce *GCECloud) GetRegionAddress(name, region string) (*compute.Address, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newAddressMetricContext("get", region)
	v, err := gce.c.Addresses().Get(ctx, meta.RegionalKey(name, region))
	return v, mc.Observe(err)
}

// GetAlphaRegionAddress returns the Alpha, regional address by name.
func (gce *GCECloud) GetAlphaRegionAddress(name, region string) (*computealpha.Address, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newAddressMetricContext("get", region)
	v, err := gce.c.AlphaAddresses().Get(ctx, meta.RegionalKey(name, region))
	return v, mc.Observe(err)
}

// GetBetaRegionAddress returns the beta region address by name
func (gce *GCECloud) GetBetaRegionAddress(name, region string) (*computebeta.Address, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newAddressMetricContext("get", region)
	v, err := gce.c.BetaAddresses().Get(ctx, meta.RegionalKey(name, region))
	return v, mc.Observe(err)
}

// GetRegionAddressByIP returns the regional address matching the given IP address.
func (gce *GCECloud) GetRegionAddressByIP(region, ipAddress string) (*compute.Address, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newAddressMetricContext("list", region)
	addrs, err := gce.c.Addresses().List(ctx, region, filter.Regexp("address", ipAddress))

	mc.Observe(err)
	if err != nil {
		return nil, err
	}

	if len(addrs) > 1 {
		glog.Warningf("More than one addresses matching the IP %q: %v", ipAddress, addrNames(addrs))
	}
	for _, addr := range addrs {
		if addr.Address == ipAddress {
			return addr, nil
		}
	}
	return nil, makeGoogleAPINotFoundError(fmt.Sprintf("Address with IP %q was not found in region %q", ipAddress, region))
}

// GetBetaRegionAddressByIP returns the beta regional address matching the given IP address.
func (gce *GCECloud) GetBetaRegionAddressByIP(region, ipAddress string) (*computebeta.Address, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newAddressMetricContext("list", region)
	addrs, err := gce.c.BetaAddresses().List(ctx, region, filter.Regexp("address", ipAddress))

	mc.Observe(err)
	if err != nil {
		return nil, err
	}

	if len(addrs) > 1 {
		glog.Warningf("More than one addresses matching the IP %q: %v", ipAddress, addrNames(addrs))
	}
	for _, addr := range addrs {
		if addr.Address == ipAddress {
			return addr, nil
		}
	}
	return nil, makeGoogleAPINotFoundError(fmt.Sprintf("Address with IP %q was not found in region %q", ipAddress, region))
}

// TODO(#51665): retire this function once Network Tiers becomes Beta in GCP.
func (gce *GCECloud) getNetworkTierFromAddress(name, region string) (string, error) {
	if !gce.AlphaFeatureGate.Enabled(AlphaFeatureNetworkTiers) {
		return cloud.NetworkTierDefault.ToGCEValue(), nil
	}
	addr, err := gce.GetAlphaRegionAddress(name, region)
	if err != nil {
		return handleAlphaNetworkTierGetError(err)
	}
	return addr.NetworkTier, nil
}

func addrNames(items interface{}) []string {
	var ret []string
	switch items := items.(type) {
	case []compute.Address:
		for _, a := range items {
			ret = append(ret, a.Name)
		}
	case []computebeta.Address:
		for _, a := range items {
			ret = append(ret, a.Name)
		}
	}
	return ret
}
