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

package policy

import (
	"strings"

	"k8s.io/apiserver/pkg/apis/audit"
	auditinternal "k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

const (
	// DefaultAuditLevel is the default level to audit at, if no policy rules are matched.
	DefaultAuditLevel = audit.LevelNone
)

// NewPolicyRuleEvaluator creates a new policy rule evaluator.
func NewPolicyRuleEvaluator(policy *audit.Policy) auditinternal.PolicyRuleEvaluator {
	for i, rule := range policy.Rules {
		policy.Rules[i].OmitStages = unionStages(policy.OmitStages, rule.OmitStages)
	}
	return &policyRuleEvaluator{*policy}
}

func unionStages(stageLists ...[]audit.Stage) []audit.Stage {
	m := make(map[audit.Stage]bool)
	for _, sl := range stageLists {
		for _, s := range sl {
			m[s] = true
		}
	}
	result := make([]audit.Stage, 0, len(m))
	for key := range m {
		result = append(result, key)
	}
	return result
}

// NewFakePolicyRuleEvaluator creates a fake policy rule evaluator that returns
// a constant level for all requests (for testing).
func NewFakePolicyRuleEvaluator(level audit.Level, stage []audit.Stage) auditinternal.PolicyRuleEvaluator {
	return &fakePolicyRuleEvaluator{level, stage}
}

type policyRuleEvaluator struct {
	audit.Policy
}

func (p *policyRuleEvaluator) EvaluatePolicyRule(attrs authorizer.Attributes) auditinternal.RequestAuditConfig {
	for _, rule := range p.Rules {
		if ruleMatches(&rule, attrs) {
			return auditinternal.RequestAuditConfig{
				Level:             rule.Level,
				OmitStages:        rule.OmitStages,
				OmitManagedFields: isOmitManagedFields(&rule, p.OmitManagedFields),
			}
		}
	}

	return auditinternal.RequestAuditConfig{
		Level:             DefaultAuditLevel,
		OmitStages:        p.OmitStages,
		OmitManagedFields: p.OmitManagedFields,
	}
}

// isOmitManagedFields returns whether to omit managed fields from the request
// and response bodies from being written to the API audit log.
// If a user specifies OmitManagedFields inside a policy rule, that overrides
// the global policy default in Policy.OmitManagedFields.
func isOmitManagedFields(policyRule *audit.PolicyRule, policyDefault bool) bool {
	if policyRule.OmitManagedFields == nil {
		return policyDefault
	}

	return *policyRule.OmitManagedFields
}

// Check whether the rule matches the request attrs.
func ruleMatches(r *audit.PolicyRule, attrs authorizer.Attributes) bool {
	user := attrs.GetUser()
	if len(r.Users) > 0 {
		if user == nil || !hasString(r.Users, user.GetName()) {
			return false
		}
	}
	if len(r.UserGroups) > 0 {
		if user == nil {
			return false
		}
		matched := false
		for _, group := range user.GetGroups() {
			if hasString(r.UserGroups, group) {
				matched = true
				break
			}
		}
		if !matched {
			return false
		}
	}
	if len(r.Verbs) > 0 {
		if !hasString(r.Verbs, attrs.GetVerb()) {
			return false
		}
	}

	if len(r.Namespaces) > 0 || len(r.Resources) > 0 {
		return ruleMatchesResource(r, attrs)
	}

	if len(r.NonResourceURLs) > 0 {
		return ruleMatchesNonResource(r, attrs)
	}

	return true
}

// Check whether the rule's non-resource URLs match the request attrs.
func ruleMatchesNonResource(r *audit.PolicyRule, attrs authorizer.Attributes) bool {
	if attrs.IsResourceRequest() {
		return false
	}

	path := attrs.GetPath()
	for _, spec := range r.NonResourceURLs {
		if pathMatches(path, spec) {
			return true
		}
	}

	return false
}

// Check whether the path matches the path specification.
func pathMatches(path, spec string) bool {
	// Allow wildcard match
	if spec == "*" {
		return true
	}
	// Allow exact match
	if spec == path {
		return true
	}
	// Allow a trailing * subpath match
	if strings.HasSuffix(spec, "*") && strings.HasPrefix(path, strings.TrimRight(spec, "*")) {
		return true
	}
	return false
}

// Check whether the rule's resource fields match the request attrs.
func ruleMatchesResource(r *audit.PolicyRule, attrs authorizer.Attributes) bool {
	if !attrs.IsResourceRequest() {
		return false
	}

	if len(r.Namespaces) > 0 {
		if !hasString(r.Namespaces, attrs.GetNamespace()) { // Non-namespaced resources use the empty string.
			return false
		}
	}
	if len(r.Resources) == 0 {
		return true
	}

	apiGroup := attrs.GetAPIGroup()
	resource := attrs.GetResource()
	subresource := attrs.GetSubresource()
	combinedResource := resource
	// If subresource, the resource in the policy must match "(resource)/(subresource)"
	if subresource != "" {
		combinedResource = resource + "/" + subresource
	}

	name := attrs.GetName()

	for _, gr := range r.Resources {
		if gr.Group == apiGroup {
			if len(gr.Resources) == 0 {
				return true
			}
			for _, res := range gr.Resources {
				if len(gr.ResourceNames) == 0 || hasString(gr.ResourceNames, name) {
					// match "*"
					if res == combinedResource || res == "*" {
						return true
					}
					// match "*/subresource"
					if len(subresource) > 0 && strings.HasPrefix(res, "*/") && subresource == strings.TrimPrefix(res, "*/") {
						return true
					}
					// match "resource/*"
					if strings.HasSuffix(res, "/*") && resource == strings.TrimSuffix(res, "/*") {
						return true
					}
				}
			}
		}
	}
	return false
}

// Utility function to check whether a string slice contains a string.
func hasString(slice []string, value string) bool {
	for _, s := range slice {
		if s == value {
			return true
		}
	}
	return false
}

type fakePolicyRuleEvaluator struct {
	level audit.Level
	stage []audit.Stage
}

func (f *fakePolicyRuleEvaluator) EvaluatePolicyRule(_ authorizer.Attributes) auditinternal.RequestAuditConfig {
	return auditinternal.RequestAuditConfig{
		Level:      f.level,
		OmitStages: f.stage,
	}
}
