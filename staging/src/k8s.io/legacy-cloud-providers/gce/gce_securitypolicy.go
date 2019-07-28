/*
Copyright 2018 The Kubernetes Authors.

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
	computebeta "google.golang.org/api/compute/v0.beta"

	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/filter"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/meta"
)

func newSecurityPolicyMetricContextWithVersion(request, version string) *metricContext {
	return newGenericMetricContext("securitypolicy", request, "", unusedMetricLabel, version)
}

// GetBetaSecurityPolicy retrieves a security policy.
func (g *Cloud) GetBetaSecurityPolicy(name string) (*computebeta.SecurityPolicy, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newSecurityPolicyMetricContextWithVersion("get", computeBetaVersion)
	v, err := g.c.BetaSecurityPolicies().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// ListBetaSecurityPolicy lists all security policies in the project.
func (g *Cloud) ListBetaSecurityPolicy() ([]*computebeta.SecurityPolicy, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newSecurityPolicyMetricContextWithVersion("list", computeBetaVersion)
	v, err := g.c.BetaSecurityPolicies().List(ctx, filter.None)
	return v, mc.Observe(err)
}

// CreateBetaSecurityPolicy creates the given security policy.
func (g *Cloud) CreateBetaSecurityPolicy(sp *computebeta.SecurityPolicy) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newSecurityPolicyMetricContextWithVersion("create", computeBetaVersion)
	return mc.Observe(g.c.BetaSecurityPolicies().Insert(ctx, meta.GlobalKey(sp.Name), sp))
}

// DeleteBetaSecurityPolicy deletes the given security policy.
func (g *Cloud) DeleteBetaSecurityPolicy(name string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newSecurityPolicyMetricContextWithVersion("delete", computeBetaVersion)
	return mc.Observe(g.c.BetaSecurityPolicies().Delete(ctx, meta.GlobalKey(name)))
}

// PatchBetaSecurityPolicy applies the given security policy as a
// patch to an existing security policy.
func (g *Cloud) PatchBetaSecurityPolicy(sp *computebeta.SecurityPolicy) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newSecurityPolicyMetricContextWithVersion("patch", computeBetaVersion)
	return mc.Observe(g.c.BetaSecurityPolicies().Patch(ctx, meta.GlobalKey(sp.Name), sp))
}

// GetRuleForBetaSecurityPolicy gets rule from a security policy.
func (g *Cloud) GetRuleForBetaSecurityPolicy(name string) (*computebeta.SecurityPolicyRule, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newSecurityPolicyMetricContextWithVersion("get_rule", computeBetaVersion)
	v, err := g.c.BetaSecurityPolicies().GetRule(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// AddRuletoBetaSecurityPolicy adds the given security policy rule to
// a security policy.
func (g *Cloud) AddRuletoBetaSecurityPolicy(name string, spr *computebeta.SecurityPolicyRule) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newSecurityPolicyMetricContextWithVersion("add_rule", computeBetaVersion)
	return mc.Observe(g.c.BetaSecurityPolicies().AddRule(ctx, meta.GlobalKey(name), spr))
}

// PatchRuleForBetaSecurityPolicy patches the given security policy
// rule to a security policy.
func (g *Cloud) PatchRuleForBetaSecurityPolicy(name string, spr *computebeta.SecurityPolicyRule) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newSecurityPolicyMetricContextWithVersion("patch_rule", computeBetaVersion)
	return mc.Observe(g.c.BetaSecurityPolicies().PatchRule(ctx, meta.GlobalKey(name), spr))
}

// RemoveRuleFromBetaSecurityPolicy removes rule from a security policy.
func (g *Cloud) RemoveRuleFromBetaSecurityPolicy(name string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newSecurityPolicyMetricContextWithVersion("remove_rule", computeBetaVersion)
	return mc.Observe(g.c.BetaSecurityPolicies().RemoveRule(ctx, meta.GlobalKey(name)))
}
