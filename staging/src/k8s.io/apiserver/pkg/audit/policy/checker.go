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
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

const (
	// DefaultAuditLevel is the default level to audit at, if no policy rules are matched.
	DefaultAuditLevel = audit.LevelNone
)

// Checker exposes methods for checking the policy rules.
type Checker interface {
	// Check the audit level for a request with the given authorizer attributes.
	LevelAndStages(authorizer.Attributes) (audit.Level, []audit.Stage)
}

// NewChecker creates a new policy checker.
func NewChecker(policy *audit.Policy) Checker {
	for i, rule := range policy.Rules {
		policy.Rules[i].OmitStages = unionStages(policy.OmitStages, rule.OmitStages)
	}
	return &policyChecker{*policy}
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

// FakeChecker creates a checker that returns a constant level for all requests (for testing).
func FakeChecker(level audit.Level, stage []audit.Stage) Checker {
	return &fakeChecker{level, stage}
}

type policyChecker struct {
	audit.Policy
}

func (p *policyChecker) LevelAndStages(attrs authorizer.Attributes) (audit.Level, []audit.Stage) {
	for _, rule := range p.Rules {
		if ruleMatches(&rule, attrs) {
			return rule.Level, rule.OmitStages
		}
	}
	return DefaultAuditLevel, p.OmitStages
}

// Check whether the rule matches the request attrs.
func ruleMatches(r *audit.PolicyRule, attrs authorizer.Attributes) bool {
	if len(r.Users) > 0 && attrs.GetUser() != nil {
		if !hasString(r.Users, attrs.GetUser().GetName()) {
			return false
		}
	}
	if len(r.UserGroups) > 0 && attrs.GetUser() != nil {
		matched := false
		for _, group := range attrs.GetUser().GetGroups() {
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
	// If subresource, the resource in the policy must match "(resource)/(subresource)"
	//
	// TODO: consider adding options like "pods/*" to match all subresources.
	if sr := attrs.GetSubresource(); sr != "" {
		resource = resource + "/" + sr
	}

	name := attrs.GetName()

	for _, gr := range r.Resources {
		if gr.Group == apiGroup {
			if len(gr.Resources) == 0 {
				return true
			}
			for _, res := range gr.Resources {
				if res == resource {
					if len(gr.ResourceNames) == 0 || hasString(gr.ResourceNames, name) {
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

type fakeChecker struct {
	level audit.Level
	stage []audit.Stage
}

func (f *fakeChecker) LevelAndStages(_ authorizer.Attributes) (audit.Level, []audit.Stage) {
	return f.level, f.stage
}
