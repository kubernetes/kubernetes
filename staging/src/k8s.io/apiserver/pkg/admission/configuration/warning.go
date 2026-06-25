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
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
)

// excludedWebhookResources is the set of virtual auth/authz resources that admission webhooks
// do not intercept when the ExcludeAdmissionWebhookVirtualResources feature is enabled.
//
// NOTE: This list MUST be kept in sync with the authoritative list in
// k8s.io/kubernetes/pkg/kubeapiserver/admission/exclusion (exclusion.Excluded()). This package
// lives in staging and cannot import from k8s.io/kubernetes, so the list is duplicated here.
// This copy is only used for the advisory log below; the dispatch path uses the authoritative
// list injected via SetExcludedAdmissionResources.
var excludedWebhookResources = sets.New(
	schema.GroupResource{Group: "authentication.k8s.io", Resource: "selfsubjectreviews"},
	schema.GroupResource{Group: "authentication.k8s.io", Resource: "tokenreviews"},
	schema.GroupResource{Group: "authorization.k8s.io", Resource: "localsubjectaccessreviews"},
	schema.GroupResource{Group: "authorization.k8s.io", Resource: "selfsubjectaccessreviews"},
	schema.GroupResource{Group: "authorization.k8s.io", Resource: "selfsubjectrulesreviews"},
	schema.GroupResource{Group: "authorization.k8s.io", Resource: "subjectaccessreviews"},
)

// logExcludedResourcesForValidatingWebhook logs an advisory for a ValidatingWebhookConfiguration
// whose rules name resources excluded from admission webhooks. It is a no-op when
// ExcludeAdmissionWebhookVirtualResources is disabled.
func logExcludedResourcesForValidatingWebhook(name string, webhooks []v1.ValidatingWebhook) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.ExcludeAdmissionWebhookVirtualResources) {
		return
	}
	var rules []v1.RuleWithOperations
	for _, w := range webhooks {
		rules = append(rules, w.Rules...)
	}
	logExcludedWebhookResources("ValidatingWebhookConfiguration", name, rules)
}

// logExcludedResourcesForMutatingWebhook logs an advisory for a MutatingWebhookConfiguration
// whose rules name resources excluded from admission webhooks. It is a no-op when
// ExcludeAdmissionWebhookVirtualResources is disabled.
func logExcludedResourcesForMutatingWebhook(name string, webhooks []v1.MutatingWebhook) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.ExcludeAdmissionWebhookVirtualResources) {
		return
	}
	var rules []v1.RuleWithOperations
	for _, w := range webhooks {
		rules = append(rules, w.Rules...)
	}
	logExcludedWebhookResources("MutatingWebhookConfiguration", name, rules)
}

func logExcludedWebhookResources(kind, name string, rules []v1.RuleWithOperations) {
	excluded := sets.New[schema.GroupResource]()
	for _, r := range rules {
		excluded.Insert(excludedResourcesNamedByRule(r.APIGroups, r.APIVersions, r.Resources)...)
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
// A rule that uses a wildcard ("*") in apiGroups, apiVersions, or resources is not flagged
// because its intent toward the excluded resource is ambiguous.
func excludedResourcesNamedByRule(apiGroups, apiVersions, resources []string) []schema.GroupResource {
	if slices.Contains(apiGroups, "*") || slices.Contains(apiVersions, "*") || slices.Contains(resources, "*") {
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
