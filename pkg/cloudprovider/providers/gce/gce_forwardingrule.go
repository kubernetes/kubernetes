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
	"context"

	computealpha "google.golang.org/api/compute/v0.alpha"
	compute "google.golang.org/api/compute/v1"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/filter"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
)

func newForwardingRuleMetricContext(request, region string) *metricContext {
	return newForwardingRuleMetricContextWithVersion(request, region, computeV1Version)
}
func newForwardingRuleMetricContextWithVersion(request, region, version string) *metricContext {
	return newGenericMetricContext("forwardingrule", request, region, unusedMetricLabel, version)
}

// CreateGlobalForwardingRule creates the passed GlobalForwardingRule
func (gce *GCECloud) CreateGlobalForwardingRule(rule *compute.ForwardingRule) error {
	mc := newForwardingRuleMetricContext("create", "")
	return mc.Observe(gce.c.GlobalForwardingRules().Insert(context.Background(), meta.GlobalKey(rule.Name), rule))
}

// SetProxyForGlobalForwardingRule links the given TargetHttp(s)Proxy with the given GlobalForwardingRule.
// targetProxyLink is the SelfLink of a TargetHttp(s)Proxy.
func (gce *GCECloud) SetProxyForGlobalForwardingRule(forwardingRuleName, targetProxyLink string) error {
	mc := newForwardingRuleMetricContext("set_proxy", "")
	target := &compute.TargetReference{Target: targetProxyLink}
	return mc.Observe(gce.c.GlobalForwardingRules().SetTarget(context.Background(), meta.GlobalKey(forwardingRuleName), target))
}

// DeleteGlobalForwardingRule deletes the GlobalForwardingRule by name.
func (gce *GCECloud) DeleteGlobalForwardingRule(name string) error {
	mc := newForwardingRuleMetricContext("delete", "")
	return mc.Observe(gce.c.GlobalForwardingRules().Delete(context.Background(), meta.GlobalKey(name)))
}

// GetGlobalForwardingRule returns the GlobalForwardingRule by name.
func (gce *GCECloud) GetGlobalForwardingRule(name string) (*compute.ForwardingRule, error) {
	mc := newForwardingRuleMetricContext("get", "")
	v, err := gce.c.GlobalForwardingRules().Get(context.Background(), meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// ListGlobalForwardingRules lists all GlobalForwardingRules in the project.
func (gce *GCECloud) ListGlobalForwardingRules() ([]*compute.ForwardingRule, error) {
	mc := newForwardingRuleMetricContext("list", "")
	v, err := gce.c.GlobalForwardingRules().List(context.Background(), filter.None)
	return v, mc.Observe(err)
}

// GetRegionForwardingRule returns the RegionalForwardingRule by name & region.
func (gce *GCECloud) GetRegionForwardingRule(name, region string) (*compute.ForwardingRule, error) {
	mc := newForwardingRuleMetricContext("get", region)
	v, err := gce.c.ForwardingRules().Get(context.Background(), meta.RegionalKey(name, region))
	return v, mc.Observe(err)
}

// GetAlphaRegionForwardingRule returns the Alpha forwarding rule by name & region.
func (gce *GCECloud) GetAlphaRegionForwardingRule(name, region string) (*computealpha.ForwardingRule, error) {
	mc := newForwardingRuleMetricContextWithVersion("get", region, computeAlphaVersion)
	v, err := gce.c.AlphaForwardingRules().Get(context.Background(), meta.RegionalKey(name, region))
	return v, mc.Observe(err)
}

// ListRegionForwardingRules lists all RegionalForwardingRules in the project & region.
func (gce *GCECloud) ListRegionForwardingRules(region string) ([]*compute.ForwardingRule, error) {
	mc := newForwardingRuleMetricContext("list", region)
	v, err := gce.c.ForwardingRules().List(context.Background(), region, filter.None)
	return v, mc.Observe(err)
}

// ListAlphaRegionForwardingRules lists all RegionalForwardingRules in the project & region.
func (gce *GCECloud) ListAlphaRegionForwardingRules(region string) ([]*computealpha.ForwardingRule, error) {
	mc := newForwardingRuleMetricContextWithVersion("list", region, computeAlphaVersion)
	v, err := gce.c.AlphaForwardingRules().List(context.Background(), region, filter.None)
	return v, mc.Observe(err)
}

// CreateRegionForwardingRule creates and returns a
// RegionalForwardingRule that points to the given BackendService
func (gce *GCECloud) CreateRegionForwardingRule(rule *compute.ForwardingRule, region string) error {
	mc := newForwardingRuleMetricContext("create", region)
	return mc.Observe(gce.c.ForwardingRules().Insert(context.Background(), meta.RegionalKey(rule.Name, region), rule))
}

// CreateAlphaRegionForwardingRule creates and returns an Alpha
// forwarding fule in the given region.
func (gce *GCECloud) CreateAlphaRegionForwardingRule(rule *computealpha.ForwardingRule, region string) error {
	mc := newForwardingRuleMetricContextWithVersion("create", region, computeAlphaVersion)
	return mc.Observe(gce.c.AlphaForwardingRules().Insert(context.Background(), meta.RegionalKey(rule.Name, region), rule))
}

// DeleteRegionForwardingRule deletes the RegionalForwardingRule by name & region.
func (gce *GCECloud) DeleteRegionForwardingRule(name, region string) error {
	mc := newForwardingRuleMetricContext("delete", region)
	return mc.Observe(gce.c.ForwardingRules().Delete(context.Background(), meta.RegionalKey(name, region)))
}

// TODO(#51665): retire this function once Network Tiers becomes Beta in GCP.
func (gce *GCECloud) getNetworkTierFromForwardingRule(name, region string) (string, error) {
	if !gce.AlphaFeatureGate.Enabled(AlphaFeatureNetworkTiers) {
		return NetworkTierDefault.ToGCEValue(), nil
	}
	fwdRule, err := gce.GetAlphaRegionForwardingRule(name, region)
	if err != nil {
		return handleAlphaNetworkTierGetError(err)
	}
	return fwdRule.NetworkTier, nil
}
