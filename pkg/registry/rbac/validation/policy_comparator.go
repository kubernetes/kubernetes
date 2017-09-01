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

package validation

import (
	"strings"

	"k8s.io/kubernetes/pkg/apis/rbac"
)

// Covers determines whether or not the ownerRules cover the servantRules in terms of allowed actions.
// It returns whether or not the ownerRules cover and a list of the rules that the ownerRules do not cover.
func Covers(ownerRules, servantRules []rbac.PolicyRule) (bool, []rbac.PolicyRule) {
	// 1.  Break every servantRule into individual rule tuples: group, verb, resource, resourceName
	// 2.  Compare the mini-rules against each owner rule.  Because the breakdown is down to the most atomic level, we're guaranteed that each mini-servant rule will be either fully covered or not covered by a single owner rule
	// 3.  Any left over mini-rules means that we are not covered and we have a nice list of them.
	// TODO: it might be nice to collapse the list down into something more human readable

	subrules := []rbac.PolicyRule{}
	for _, servantRule := range servantRules {
		subrules = append(subrules, BreakdownRule(servantRule)...)
	}

	uncoveredRules := []rbac.PolicyRule{}
	for _, subrule := range subrules {
		covered := false
		for _, ownerRule := range ownerRules {
			if ruleCovers(ownerRule, subrule) {
				covered = true
				break
			}
		}

		if !covered {
			uncoveredRules = append(uncoveredRules, subrule)
		}
	}

	return (len(uncoveredRules) == 0), uncoveredRules
}

// BreadownRule takes a rule and builds an equivalent list of rules that each have at most one verb, one
// resource, and one resource name
func BreakdownRule(rule rbac.PolicyRule) []rbac.PolicyRule {
	subrules := []rbac.PolicyRule{}
	for _, group := range rule.APIGroups {
		for _, resource := range rule.Resources {
			for _, verb := range rule.Verbs {
				if len(rule.ResourceNames) > 0 {
					for _, resourceName := range rule.ResourceNames {
						subrules = append(subrules, rbac.PolicyRule{APIGroups: []string{group}, Resources: []string{resource}, Verbs: []string{verb}, ResourceNames: []string{resourceName}})
					}

				} else {
					subrules = append(subrules, rbac.PolicyRule{APIGroups: []string{group}, Resources: []string{resource}, Verbs: []string{verb}})
				}

			}
		}
	}

	// Non-resource URLs are unique because they only combine with verbs.
	for _, nonResourceURL := range rule.NonResourceURLs {
		for _, verb := range rule.Verbs {
			subrules = append(subrules, rbac.PolicyRule{NonResourceURLs: []string{nonResourceURL}, Verbs: []string{verb}})
		}
	}

	return subrules
}

func has(set []string, ele string) bool {
	for _, s := range set {
		if s == ele {
			return true
		}
	}
	return false
}

func hasAll(set, contains []string) bool {
	owning := make(map[string]struct{}, len(set))
	for _, ele := range set {
		owning[ele] = struct{}{}
	}
	for _, ele := range contains {
		if _, ok := owning[ele]; !ok {
			return false
		}
	}
	return true
}

func nonResourceURLsCoversAll(set, covers []string) bool {
	for _, path := range covers {
		covered := false
		for _, owner := range set {
			if nonResourceURLCovers(owner, path) {
				covered = true
				break
			}
		}
		if !covered {
			return false
		}
	}
	return true
}

func nonResourceURLCovers(ownerPath, subPath string) bool {
	if ownerPath == subPath {
		return true
	}
	return strings.HasSuffix(ownerPath, "*") && strings.HasPrefix(subPath, strings.TrimRight(ownerPath, "*"))
}

// ruleCovers determines whether the ownerRule (which may have multiple verbs, resources, and resourceNames) covers
// the subrule (which may only contain at most one verb, resource, and resourceName)
func ruleCovers(ownerRule, subRule rbac.PolicyRule) bool {
	verbMatches := has(ownerRule.Verbs, rbac.VerbAll) || hasAll(ownerRule.Verbs, subRule.Verbs)
	groupMatches := has(ownerRule.APIGroups, rbac.APIGroupAll) || hasAll(ownerRule.APIGroups, subRule.APIGroups)
	resourceMatches := has(ownerRule.Resources, rbac.ResourceAll) || hasAll(ownerRule.Resources, subRule.Resources)
	nonResourceURLMatches := nonResourceURLsCoversAll(ownerRule.NonResourceURLs, subRule.NonResourceURLs)

	resourceNameMatches := false

	if len(subRule.ResourceNames) == 0 {
		resourceNameMatches = (len(ownerRule.ResourceNames) == 0)
	} else {
		resourceNameMatches = (len(ownerRule.ResourceNames) == 0) || hasAll(ownerRule.ResourceNames, subRule.ResourceNames)
	}

	return verbMatches && groupMatches && resourceMatches && resourceNameMatches && nonResourceURLMatches
}
