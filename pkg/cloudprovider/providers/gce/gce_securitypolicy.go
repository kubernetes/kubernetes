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
	"context"

	computebeta "google.golang.org/api/compute/v0.beta"

	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/filter"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
)

func newSecurityPolicyMetricContextWithVersion(request, version string) *metricContext {
	return newGenericMetricContext("securitypolicy", request, "", unusedMetricLabel, version)
}

// GetBetaSecurityPolicy retrieves a security policy.
func (gce *GCECloud) GetBetaSecurityPolicy(name string) (*computebeta.SecurityPolicy, error) {
	mc := newSecurityPolicyMetricContextWithVersion("get", computeBetaVersion)
	v, err := gce.c.BetaSecurityPolicies().Get(context.Background(), meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// ListBetaSecurityPolicy lists all security policies in the project.
func (gce *GCECloud) ListBetaSecurityPolicy() ([]*computebeta.SecurityPolicy, error) {
	mc := newSecurityPolicyMetricContextWithVersion("list", computeBetaVersion)
	v, err := gce.c.BetaSecurityPolicies().List(context.Background(), filter.None)
	return v, mc.Observe(err)
}

// CreateBetaSecurityPolicy creates the given security policy.
func (gce *GCECloud) CreateBetaSecurityPolicy(sp *computebeta.SecurityPolicy) error {
	mc := newSecurityPolicyMetricContextWithVersion("create", computeBetaVersion)
	return mc.Observe(gce.c.BetaSecurityPolicies().Insert(context.Background(), meta.GlobalKey(sp.Name), sp))
}

// DeleteBetaSecurityPolicy deletes the given security policy.
func (gce *GCECloud) DeleteBetaSecurityPolicy(name string) error {
	mc := newSecurityPolicyMetricContextWithVersion("delete", computeBetaVersion)
	return mc.Observe(gce.c.BetaSecurityPolicies().Delete(context.Background(), meta.GlobalKey(name)))
}

// PatchBetaSecurityPolicy applies the given security policy as a
// patch to an existing security policy.
func (gce *GCECloud) PatchBetaSecurityPolicy(sp *computebeta.SecurityPolicy) error {
	mc := newSecurityPolicyMetricContextWithVersion("patch", computeBetaVersion)
	return mc.Observe(gce.c.BetaSecurityPolicies().Patch(context.Background(), meta.GlobalKey(sp.Name), sp))
}

// GetRuleForBetaSecurityPolicy gets rule from a security policy.
func (gce *GCECloud) GetRuleForBetaSecurityPolicy(name string) (*computebeta.SecurityPolicyRule, error) {
	mc := newSecurityPolicyMetricContextWithVersion("get_rule", computeBetaVersion)
	v, err := gce.c.BetaSecurityPolicies().GetRule(context.Background(), meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// AddRuletoBetaSecurityPolicy adds the given security policy rule to
// a security policy.
func (gce *GCECloud) AddRuletoBetaSecurityPolicy(name string, spr *computebeta.SecurityPolicyRule) error {
	mc := newSecurityPolicyMetricContextWithVersion("add_rule", computeBetaVersion)
	return mc.Observe(gce.c.BetaSecurityPolicies().AddRule(context.Background(), meta.GlobalKey(name), spr))
}

// PatchRuleForBetaSecurityPolicy patches the given security policy
// rule to a security policy.
func (gce *GCECloud) PatchRuleForBetaSecurityPolicy(name string, spr *computebeta.SecurityPolicyRule) error {
	mc := newSecurityPolicyMetricContextWithVersion("patch_rule", computeBetaVersion)
	return mc.Observe(gce.c.BetaSecurityPolicies().PatchRule(context.Background(), meta.GlobalKey(name), spr))
}

// RemoveRuleFromBetaSecurityPolicy removes rule from a security policy.
func (gce *GCECloud) RemoveRuleFromBetaSecurityPolicy(name string) error {
	mc := newSecurityPolicyMetricContextWithVersion("remove_rule", computeBetaVersion)
	return mc.Observe(gce.c.BetaSecurityPolicies().RemoveRule(context.Background(), meta.GlobalKey(name)))
}
