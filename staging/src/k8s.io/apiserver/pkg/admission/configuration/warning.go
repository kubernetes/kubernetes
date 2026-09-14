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

package configuration

import (
	"slices"

	v1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
)

// logExcludedResourcesForValidatingWebhook logs an advisory for a ValidatingWebhookConfiguration whose rules name resources excluded from admission webhooks.
// It is a no-op when excludedWebhookResources is empty.
func logExcludedResourcesForValidatingWebhook(name string, webhooks []v1.ValidatingWebhook, excludedWebhookResources sets.Set[schema.GroupResource]) {
	if excludedWebhookResources.Len() == 0 {
		return
	}
	var rules []v1.RuleWithOperations
	for _, w := range webhooks {
		rules = append(rules, w.Rules...)
	}
	logExcludedWebhookResources("ValidatingWebhookConfiguration", name, rules, excludedWebhookResources)
}

// logExcludedResourcesForMutatingWebhook logs an advisory for a MutatingWebhookConfiguration whose rules name resources excluded from admission webhooks.
// It is a no-op when excludedWebhookResources is empty.
func logExcludedResourcesForMutatingWebhook(name string, webhooks []v1.MutatingWebhook, excludedWebhookResources sets.Set[schema.GroupResource]) {
	if excludedWebhookResources.Len() == 0 {
		return
	}
	var rules []v1.RuleWithOperations
	for _, w := range webhooks {
		rules = append(rules, w.Rules...)
	}
	logExcludedWebhookResources("MutatingWebhookConfiguration", name, rules, excludedWebhookResources)
}

func logExcludedWebhookResources(kind, name string, rules []v1.RuleWithOperations, excludedWebhookResources sets.Set[schema.GroupResource]) {
	excluded := sets.New[schema.GroupResource]()
	for _, r := range rules {
		excluded.Insert(excludedResourcesNamedByRule(r.APIGroups, r.Resources, excludedWebhookResources)...)
	}
	if excluded.Len() == 0 {
		return
	}
	resources := make([]string, 0, excluded.Len())
	for gr := range excluded {
		resources = append(resources, gr.String())
	}
	slices.Sort(resources)
	klog.InfoS("Admission webhook configuration names resources that are excluded from admission webhooks; the matching rules will have no effect",
		"kind", kind, "name", name, "excludedResources", resources)
}

// excludedResourcesNamedByRule returns the excluded GroupResources a rule explicitly names.
// A rule that uses a wildcard ("*") in apiGroups or resources is not flagged because its intent
// toward the excluded resource is ambiguous. apiVersions are not considered because all versions
// of an excluded resource are excluded.
func excludedResourcesNamedByRule(apiGroups, resources []string, excludedWebhookResources sets.Set[schema.GroupResource]) []schema.GroupResource {
	if slices.Contains(apiGroups, "*") || slices.Contains(resources, "*") {
		return nil
	}
	var matched []schema.GroupResource
	for _, group := range apiGroups {
		for _, resource := range resources {
			gr := schema.GroupResource{Group: group, Resource: resource}
			if excludedWebhookResources.Has(gr) {
				matched = append(matched, gr)
			}
		}
	}
	return matched
}
