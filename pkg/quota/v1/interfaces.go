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

package quota

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/tools/cache"
)

// UsageStatsOptions is an options structs that describes how stats should be calculated
type UsageStatsOptions struct {
	// Namespace where stats should be calculate
	Namespace string
	// Scopes that must match counted objects
	Scopes []corev1.ResourceQuotaScope
	// Resources are the set of resources to include in the measurement
	Resources     []corev1.ResourceName
	ScopeSelector *corev1.ScopeSelector
}

// UsageStats is result of measuring observed resource use in the system
type UsageStats struct {
	// Used maps resource to quantity used
	Used corev1.ResourceList
}

// Evaluator knows how to evaluate quota usage for a particular group resource
type Evaluator interface {
	// Constraints ensures that each required resource is present on item
	Constraints(required []corev1.ResourceName, item runtime.Object) error
	// GroupResource returns the groupResource that this object knows how to evaluate
	GroupResource() schema.GroupResource
	// Handles determines if quota could be impacted by the specified attribute.
	// If true, admission control must perform quota processing for the operation, otherwise it is safe to ignore quota.
	Handles(operation admission.Attributes) bool
	// Matches returns true if the specified quota matches the input item
	Matches(resourceQuota *corev1.ResourceQuota, item runtime.Object) (bool, error)
	// MatchingScopes takes the input specified list of scopes and input object and returns the set of scopes that matches input object.
	MatchingScopes(item runtime.Object, scopes []corev1.ScopedResourceSelectorRequirement) ([]corev1.ScopedResourceSelectorRequirement, error)
	// UncoveredQuotaScopes takes the input matched scopes which are limited by configuration and the matched quota scopes. It returns the scopes which are in limited scopes but dont have a corresponding covering quota scope
	UncoveredQuotaScopes(limitedScopes []corev1.ScopedResourceSelectorRequirement, matchedQuotaScopes []corev1.ScopedResourceSelectorRequirement) ([]corev1.ScopedResourceSelectorRequirement, error)
	// MatchingResources takes the input specified list of resources and returns the set of resources evaluator matches.
	MatchingResources(input []corev1.ResourceName) []corev1.ResourceName
	// Usage returns the resource usage for the specified object
	Usage(item runtime.Object) (corev1.ResourceList, error)
	// UsageStats calculates latest observed usage stats for all objects
	UsageStats(options UsageStatsOptions) (UsageStats, error)
}

// Configuration defines how the quota system is configured.
type Configuration interface {
	// IgnoredResources are ignored by quota.
	IgnoredResources() map[schema.GroupResource]struct{}
	// Evaluators for quota evaluation.
	Evaluators() []Evaluator
}

// Registry maintains a list of evaluators
type Registry interface {
	// Add to registry
	Add(e Evaluator)
	// Remove from registry
	Remove(e Evaluator)
	// Get by group resource
	Get(gr schema.GroupResource) Evaluator
	// List from registry
	List() []Evaluator
}

// ListerForResourceFunc knows how to get a lister for a specific resource
type ListerForResourceFunc func(schema.GroupVersionResource) (cache.GenericLister, error)
