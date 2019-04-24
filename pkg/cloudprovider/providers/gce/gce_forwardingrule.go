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
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/filter"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/meta"
	computebeta "google.golang.org/api/compute/v0.beta"
	compute "google.golang.org/api/compute/v1"
)

func newForwardingRuleMetricContext(request, region string) *metricContext {
	return newForwardingRuleMetricContextWithVersion(request, region, computeV1Version)
}
func newForwardingRuleMetricContextWithVersion(request, region, version string) *metricContext {
	return newGenericMetricContext("forwardingrule", request, region, unusedMetricLabel, version)
}

// CreateGlobalForwardingRule creates the passed GlobalForwardingRule
func (g *Cloud) CreateGlobalForwardingRule(rule *compute.ForwardingRule) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newForwardingRuleMetricContext("create", "")
	return mc.Observe(g.c.GlobalForwardingRules().Insert(ctx, meta.GlobalKey(rule.Name), rule))
}

// SetProxyForGlobalForwardingRule links the given TargetHttp(s)Proxy with the given GlobalForwardingRule.
// targetProxyLink is the SelfLink of a TargetHttp(s)Proxy.
func (g *Cloud) SetProxyForGlobalForwardingRule(forwardingRuleName, targetProxyLink string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newForwardingRuleMetricContext("set_proxy", "")
	target := &compute.TargetReference{Target: targetProxyLink}
	return mc.Observe(g.c.GlobalForwardingRules().SetTarget(ctx, meta.GlobalKey(forwardingRuleName), target))
}

// DeleteGlobalForwardingRule deletes the GlobalForwardingRule by name.
func (g *Cloud) DeleteGlobalForwardingRule(name string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newForwardingRuleMetricContext("delete", "")
	return mc.Observe(g.c.GlobalForwardingRules().Delete(ctx, meta.GlobalKey(name)))
}

// GetGlobalForwardingRule returns the GlobalForwardingRule by name.
func (g *Cloud) GetGlobalForwardingRule(name string) (*compute.ForwardingRule, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newForwardingRuleMetricContext("get", "")
	v, err := g.c.GlobalForwardingRules().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// ListGlobalForwardingRules lists all GlobalForwardingRules in the project.
func (g *Cloud) ListGlobalForwardingRules() ([]*compute.ForwardingRule, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newForwardingRuleMetricContext("list", "")
	v, err := g.c.GlobalForwardingRules().List(ctx, filter.None)
	return v, mc.Observe(err)
}

// GetRegionForwardingRule returns the RegionalForwardingRule by name & region.
func (g *Cloud) GetRegionForwardingRule(name, region string) (*compute.ForwardingRule, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newForwardingRuleMetricContext("get", region)
	v, err := g.c.ForwardingRules().Get(ctx, meta.RegionalKey(name, region))
	return v, mc.Observe(err)
}

// GetBetaRegionForwardingRule returns the Beta forwarding rule by name & region.
func (g *Cloud) GetBetaRegionForwardingRule(name, region string) (*computebeta.ForwardingRule, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newForwardingRuleMetricContextWithVersion("get", region, computeBetaVersion)
	v, err := g.c.BetaForwardingRules().Get(ctx, meta.RegionalKey(name, region))
	return v, mc.Observe(err)
}

// ListRegionForwardingRules lists all RegionalForwardingRules in the project & region.
func (g *Cloud) ListRegionForwardingRules(region string) ([]*compute.ForwardingRule, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newForwardingRuleMetricContext("list", region)
	v, err := g.c.ForwardingRules().List(ctx, region, filter.None)
	return v, mc.Observe(err)
}

// ListBetaRegionForwardingRules lists all RegionalForwardingRules in the project & region.
func (g *Cloud) ListBetaRegionForwardingRules(region string) ([]*computebeta.ForwardingRule, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newForwardingRuleMetricContextWithVersion("list", region, computeBetaVersion)
	v, err := g.c.BetaForwardingRules().List(ctx, region, filter.None)
	return v, mc.Observe(err)
}

// CreateRegionForwardingRule creates and returns a
// RegionalForwardingRule that points to the given BackendService
func (g *Cloud) CreateRegionForwardingRule(rule *compute.ForwardingRule, region string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newForwardingRuleMetricContext("create", region)
	return mc.Observe(g.c.ForwardingRules().Insert(ctx, meta.RegionalKey(rule.Name, region), rule))
}

// CreateBetaRegionForwardingRule creates and returns an Beta
// forwarding fule in the given region.
func (g *Cloud) CreateBetaRegionForwardingRule(rule *computebeta.ForwardingRule, region string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newForwardingRuleMetricContextWithVersion("create", region, computeBetaVersion)
	return mc.Observe(g.c.BetaForwardingRules().Insert(ctx, meta.RegionalKey(rule.Name, region), rule))
}

// DeleteRegionForwardingRule deletes the RegionalForwardingRule by name & region.
func (g *Cloud) DeleteRegionForwardingRule(name, region string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newForwardingRuleMetricContext("delete", region)
	return mc.Observe(g.c.ForwardingRules().Delete(ctx, meta.RegionalKey(name, region)))
}

// TODO(#51665): retire this function once Network Tiers becomes Beta in GCP.
func (g *Cloud) getNetworkTierFromForwardingRule(name, region string) (string, error) {
	if !g.AlphaFeatureGate.Enabled(AlphaFeatureNetworkTiers) {
		return cloud.NetworkTierDefault.ToGCEValue(), nil
	}
	fwdRule, err := g.GetBetaRegionForwardingRule(name, region)
	if err != nil {
		return handleBetaNetworkTierGetError(err)
	}
	return fwdRule.NetworkTier, nil
}
