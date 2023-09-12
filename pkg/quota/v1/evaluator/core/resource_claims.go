/*
Copyright 2023 The Kubernetes Authors.

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

package core

import (
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	resourcev1alpha2 "k8s.io/api/resource/v1alpha2"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	quota "k8s.io/apiserver/pkg/quota/v1"
	"k8s.io/apiserver/pkg/quota/v1/generic"
	api "k8s.io/kubernetes/pkg/apis/resource"
	k8s_resource_v1alpha2 "k8s.io/kubernetes/pkg/apis/resource/v1alpha2"
)

// the name used for object count quota
var claimObjectCountName = generic.ObjectCountQuotaResourceNameFor(resourcev1alpha2.SchemeGroupVersion.WithResource("resourceclaims").GroupResource())

// V1ResourceByResourceClass returns a quota resource name by resource class.
func V1ResourceByResourceClass(resourceClass string) corev1.ResourceName {
	return corev1.ResourceName(resourceClass + corev1.ResourceClaimsPerClass)
}

// NewResourceClaimEvaluator returns an evaluator that can evaluate resource claims
func NewResourceClaimEvaluator(f quota.ListerForResourceFunc) quota.Evaluator {
	listFuncByNamespace := generic.ListResourceUsingListerFunc(f, resourcev1alpha2.SchemeGroupVersion.WithResource("resourceclaims"))
	claimEvaluator := &claimEvaluator{listFuncByNamespace: listFuncByNamespace}
	return claimEvaluator
}

// claimEvaluator knows how to evaluate quota usage for resource claims
type claimEvaluator struct {
	// listFuncByNamespace knows how to list resource claims
	listFuncByNamespace generic.ListFuncByNamespace
}

// Constraints verifies that all required resources are present on the item.
func (p *claimEvaluator) Constraints(required []corev1.ResourceName, item runtime.Object) error {
	// no-op for resource claims
	return nil
}

// GroupResource that this evaluator tracks
func (p *claimEvaluator) GroupResource() schema.GroupResource {
	return resourcev1alpha2.SchemeGroupVersion.WithResource("resourceclaims").GroupResource()
}

// Handles returns true if the evaluator should handle the specified operation.
func (p *claimEvaluator) Handles(a admission.Attributes) bool {
	op := a.GetOperation()
	if op == admission.Create {
		return true
	}
	if op == admission.Update {
		return true
	}
	return false
}

// Matches returns true if the evaluator matches the specified quota with the provided input item
func (p *claimEvaluator) Matches(resourceQuota *corev1.ResourceQuota, item runtime.Object) (bool, error) {
	return generic.Matches(resourceQuota, item, p.MatchingResources, generic.MatchesNoScopeFunc)
}

// MatchingScopes takes the input specified list of scopes and input object. Returns the set of scopes resource matches.
func (p *claimEvaluator) MatchingScopes(item runtime.Object, scopes []corev1.ScopedResourceSelectorRequirement) ([]corev1.ScopedResourceSelectorRequirement, error) {
	return []corev1.ScopedResourceSelectorRequirement{}, nil
}

// UncoveredQuotaScopes takes the input matched scopes which are limited by configuration and the matched quota scopes.
// It returns the scopes which are in limited scopes but don't have a corresponding covering quota scope
func (p *claimEvaluator) UncoveredQuotaScopes(limitedScopes []corev1.ScopedResourceSelectorRequirement, matchedQuotaScopes []corev1.ScopedResourceSelectorRequirement) ([]corev1.ScopedResourceSelectorRequirement, error) {
	return []corev1.ScopedResourceSelectorRequirement{}, nil
}

// MatchingResources takes the input specified list of resources and returns the set of resources it matches.
func (p *claimEvaluator) MatchingResources(items []corev1.ResourceName) []corev1.ResourceName {
	result := []corev1.ResourceName{}
	for _, item := range items {
		if item == claimObjectCountName /* object count quota fields */ ||
			item == corev1.ResourceClaims /* claim resources alias */ ||
			strings.HasSuffix(string(item), corev1.ResourceClaimsPerClass /* by resource class */) {
			result = append(result, item)
		}
	}
	return result
}

// Usage knows how to measure usage associated with item.
func (p *claimEvaluator) Usage(item runtime.Object) (corev1.ResourceList, error) {
	result := corev1.ResourceList{}
	claim, err := toExternalResourceClaimOrError(item)
	if err != nil {
		return result, err
	}

	// charge for claim
	result[corev1.ResourceClaims] = *(resource.NewQuantity(1, resource.DecimalSI))
	result[claimObjectCountName] = *(resource.NewQuantity(1, resource.DecimalSI))
	resourceClassClaim := V1ResourceByResourceClass(claim.Spec.ResourceClassName)
	result[resourceClassClaim] = *(resource.NewQuantity(1, resource.DecimalSI))

	return result, nil
}

// UsageStats calculates aggregate usage for the object.
func (p *claimEvaluator) UsageStats(options quota.UsageStatsOptions) (quota.UsageStats, error) {
	return generic.CalculateUsageStats(options, p.listFuncByNamespace, generic.MatchesNoScopeFunc, p.Usage)
}

// ensure we implement required interface
var _ quota.Evaluator = &claimEvaluator{}

func toExternalResourceClaimOrError(obj runtime.Object) (*resourcev1alpha2.ResourceClaim, error) {
	claim := &resourcev1alpha2.ResourceClaim{}
	switch t := obj.(type) {
	case *resourcev1alpha2.ResourceClaim:
		claim = t
	case *api.ResourceClaim:
		if err := k8s_resource_v1alpha2.Convert_resource_ResourceClaim_To_v1alpha2_ResourceClaim(t, claim, nil); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("expect *resource.ResourceClaim or *v1alpha2.ResourceClaim, got %v", t)
	}
	return claim, nil
}
