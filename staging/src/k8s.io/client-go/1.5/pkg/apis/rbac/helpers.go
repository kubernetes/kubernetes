/*
Copyright 2016 The Kubernetes Authors.

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

package rbac

import (
	"fmt"
	"strings"

	"k8s.io/client-go/1.5/pkg/api/unversioned"
)

func RoleRefGroupKind(roleRef RoleRef) unversioned.GroupKind {
	return unversioned.GroupKind{Group: roleRef.APIGroup, Kind: roleRef.Kind}
}

func VerbMatches(rule PolicyRule, requestedVerb string) bool {
	for _, ruleVerb := range rule.Verbs {
		if ruleVerb == VerbAll {
			return true
		}
		if ruleVerb == requestedVerb {
			return true
		}
	}

	return false
}

func APIGroupMatches(rule PolicyRule, requestedGroup string) bool {
	for _, ruleGroup := range rule.APIGroups {
		if ruleGroup == APIGroupAll {
			return true
		}
		if ruleGroup == requestedGroup {
			return true
		}
	}

	return false
}

func ResourceMatches(rule PolicyRule, requestedResource string) bool {
	for _, ruleResource := range rule.Resources {
		if ruleResource == ResourceAll {
			return true
		}
		if ruleResource == requestedResource {
			return true
		}
	}

	return false
}

func ResourceNameMatches(rule PolicyRule, requestedName string) bool {
	if len(rule.ResourceNames) == 0 {
		return true
	}

	for _, ruleName := range rule.ResourceNames {
		if ruleName == requestedName {
			return true
		}
	}

	return false
}

func NonResourceURLMatches(rule PolicyRule, requestedURL string) bool {
	for _, ruleURL := range rule.NonResourceURLs {
		if ruleURL == NonResourceAll {
			return true
		}
		if ruleURL == requestedURL {
			return true
		}
		if strings.HasSuffix(ruleURL, "*") && strings.HasPrefix(requestedURL, strings.TrimRight(ruleURL, "*")) {
			return true
		}
	}

	return false
}

// +k8s:deepcopy-gen=false
// PolicyRuleBuilder let's us attach methods.  A no-no for API types.
// We use it to construct rules in code.  It's more compact than trying to write them
// out in a literal and allows us to perform some basic checking during construction
type PolicyRuleBuilder struct {
	PolicyRule PolicyRule
}

func NewRule(verbs ...string) *PolicyRuleBuilder {
	return &PolicyRuleBuilder{
		PolicyRule: PolicyRule{Verbs: verbs},
	}
}

func (r *PolicyRuleBuilder) Groups(groups ...string) *PolicyRuleBuilder {
	r.PolicyRule.APIGroups = append(r.PolicyRule.APIGroups, groups...)
	return r
}

func (r *PolicyRuleBuilder) Resources(resources ...string) *PolicyRuleBuilder {
	r.PolicyRule.Resources = append(r.PolicyRule.Resources, resources...)
	return r
}

func (r *PolicyRuleBuilder) Names(names ...string) *PolicyRuleBuilder {
	r.PolicyRule.ResourceNames = append(r.PolicyRule.ResourceNames, names...)
	return r
}

func (r *PolicyRuleBuilder) URLs(urls ...string) *PolicyRuleBuilder {
	r.PolicyRule.NonResourceURLs = append(r.PolicyRule.NonResourceURLs, urls...)
	return r
}

func (r *PolicyRuleBuilder) RuleOrDie() PolicyRule {
	ret, err := r.Rule()
	if err != nil {
		panic(err)
	}
	return ret
}

func (r *PolicyRuleBuilder) Rule() (PolicyRule, error) {
	if len(r.PolicyRule.Verbs) == 0 {
		return PolicyRule{}, fmt.Errorf("verbs are required: %#v", r.PolicyRule)
	}

	switch {
	case len(r.PolicyRule.NonResourceURLs) > 0:
		if len(r.PolicyRule.APIGroups) != 0 || len(r.PolicyRule.Resources) != 0 || len(r.PolicyRule.ResourceNames) != 0 {
			return PolicyRule{}, fmt.Errorf("non-resource rule may not have apiGroups, resources, or resourceNames: %#v", r.PolicyRule)
		}
	case len(r.PolicyRule.Resources) > 0:
		if len(r.PolicyRule.NonResourceURLs) != 0 {
			return PolicyRule{}, fmt.Errorf("resource rule may not have nonResourceURLs: %#v", r.PolicyRule)
		}
		if len(r.PolicyRule.APIGroups) == 0 {
			// this a common bug
			return PolicyRule{}, fmt.Errorf("resource rule must have apiGroups: %#v", r.PolicyRule)
		}
	default:
		return PolicyRule{}, fmt.Errorf("a rule must have either nonResourceURLs or resources: %#v", r.PolicyRule)
	}

	return r.PolicyRule, nil
}
