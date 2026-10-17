/*
Copyright The Kubernetes Authors.

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
	"fmt"
	"slices"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
)

// WarningsForValidatingWebhookRules returns warnings for rules in a ValidatingWebhookConfiguration
// that explicitly name a resource in excludedResources (virtual resources that admission webhooks
// do not intercept). The excluded set is supplied by the caller to avoid depending on server-side
// packages from here.
func WarningsForValidatingWebhookRules(webhooks []admissionregistration.ValidatingWebhook, excludedResources []schema.GroupResource) []string {
	excluded := sets.New(excludedResources...)
	webhooksPath := field.NewPath("webhooks")
	var warnings []string
	for i := range webhooks {
		warnings = append(warnings, warningsForExcludedRules(webhooksPath.Index(i).Child("rules"), webhooks[i].Rules, excluded)...)
	}
	return warnings
}

// WarningsForMutatingWebhookRules returns warnings for rules in a MutatingWebhookConfiguration
// that explicitly name a resource in excludedResources.
func WarningsForMutatingWebhookRules(webhooks []admissionregistration.MutatingWebhook, excludedResources []schema.GroupResource) []string {
	excluded := sets.New(excludedResources...)
	webhooksPath := field.NewPath("webhooks")
	var warnings []string
	for i := range webhooks {
		warnings = append(warnings, warningsForExcludedRules(webhooksPath.Index(i).Child("rules"), webhooks[i].Rules, excluded)...)
	}
	return warnings
}

func warningsForExcludedRules(rulesPath *field.Path, rules []admissionregistration.RuleWithOperations, excluded sets.Set[schema.GroupResource]) []string {
	var warnings []string
	for i := range rules {
		for _, gr := range excludedResourcesNamedByRule(rules[i].APIGroups, rules[i].Resources, excluded) {
			warnings = append(warnings, fmt.Sprintf("%s: %s is excluded from admission webhooks; this rule will have no effect", rulesPath.Index(i), gr))
		}
	}
	return warnings
}

// excludedResourcesNamedByRule returns the excluded GroupResources a rule explicitly names.
// A rule that uses a wildcard ("*") in apiGroups or resources is not flagged because its intent
// toward the excluded resource is ambiguous. apiVersions are not considered because all versions
// of an excluded resource are excluded.
func excludedResourcesNamedByRule(apiGroups, resources []string, excluded sets.Set[schema.GroupResource]) []schema.GroupResource {
	if slices.Contains(apiGroups, "*") || slices.Contains(resources, "*") {
		return nil
	}
	// Dedupe with a set so a rule that repeats a group/resource is only reported once,
	// preserving the order the rule lists them in.
	seen := sets.New[schema.GroupResource]()
	var matched []schema.GroupResource
	for _, group := range apiGroups {
		for _, resource := range resources {
			gr := schema.GroupResource{Group: group, Resource: resource}
			if excluded.Has(gr) && !seen.Has(gr) {
				seen.Insert(gr)
				matched = append(matched, gr)
			}
		}
	}
	return matched
}
