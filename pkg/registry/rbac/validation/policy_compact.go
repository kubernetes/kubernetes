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

package validation

import (
	"reflect"

	rbacv1 "k8s.io/api/rbac/v1"
)

type simpleResource struct {
	Group             string
	Resource          string
	ResourceNameExist bool
	ResourceName      string
}

// CompactRules combines rules that contain a single APIGroup/Resource, differ only by verb, and contain no other attributes.
// this is a fast check, and works well with the decomposed "missing rules" list from a Covers check.
func CompactRules(rules []rbacv1.PolicyRule) ([]rbacv1.PolicyRule, error) {
	compacted := make([]rbacv1.PolicyRule, 0, len(rules))

	simpleRules := map[simpleResource]*rbacv1.PolicyRule{}
	for _, rule := range rules {
		if resource, isSimple := isSimpleResourceRule(&rule); isSimple {
			if existingRule, ok := simpleRules[resource]; ok {
				// Add the new verbs to the existing simple resource rule
				if existingRule.Verbs == nil {
					existingRule.Verbs = []string{}
				}
				existingRule.Verbs = append(existingRule.Verbs, rule.Verbs...)
			} else {
				// Copy the rule to accumulate matching simple resource rules into
				simpleRules[resource] = rule.DeepCopy()
			}
		} else {
			compacted = append(compacted, rule)
		}
	}

	// Once we've consolidated the simple resource rules, add them to the compacted list
	for _, simpleRule := range simpleRules {
		compacted = append(compacted, *simpleRule)
	}

	return compacted, nil
}

// isSimpleResourceRule returns true if the given rule contains verbs, a single resource, a single API group, at most one Resource Name, and no other values
func isSimpleResourceRule(rule *rbacv1.PolicyRule) (simpleResource, bool) {
	resource := simpleResource{}

	// If we have "complex" rule attributes, return early without allocations or expensive comparisons
	if len(rule.ResourceNames) > 1 || len(rule.NonResourceURLs) > 0 {
		return resource, false
	}
	// If we have multiple api groups or resources, return early
	if len(rule.APIGroups) != 1 || len(rule.Resources) != 1 {
		return resource, false
	}

	// Test if this rule only contains APIGroups/Resources/Verbs/ResourceNames
	simpleRule := &rbacv1.PolicyRule{APIGroups: rule.APIGroups, Resources: rule.Resources, Verbs: rule.Verbs, ResourceNames: rule.ResourceNames}
	if !reflect.DeepEqual(simpleRule, rule) {
		return resource, false
	}

	if len(rule.ResourceNames) == 0 {
		resource = simpleResource{Group: rule.APIGroups[0], Resource: rule.Resources[0], ResourceNameExist: false}
	} else {
		resource = simpleResource{Group: rule.APIGroups[0], Resource: rule.Resources[0], ResourceNameExist: true, ResourceName: rule.ResourceNames[0]}
	}

	return resource, true
}
