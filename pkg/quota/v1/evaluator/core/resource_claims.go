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
	resourceapi "k8s.io/api/resource/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	quota "k8s.io/apiserver/pkg/quota/v1"
	"k8s.io/apiserver/pkg/quota/v1/generic"
	resourceinternal "k8s.io/kubernetes/pkg/apis/resource"
	resourceversioned "k8s.io/kubernetes/pkg/apis/resource/v1beta1"
)

// The name used for object count quota. This evaluator takes over counting
// those because of it's GroupResource, so it has to implement this
// count.
var ClaimObjectCountName = generic.ObjectCountQuotaResourceNameFor(resourceapi.SchemeGroupVersion.WithResource("resourceclaims").GroupResource())

// V1ResourceByDeviceClass returns a quota resource name by device class.
func V1ResourceByDeviceClass(className string) corev1.ResourceName {
	return corev1.ResourceName(className + corev1.ResourceClaimsPerClass)
}

// NewResourceClaimEvaluator returns an evaluator that can evaluate resource claims
func NewResourceClaimEvaluator(f quota.ListerForResourceFunc) quota.Evaluator {
	listFuncByNamespace := generic.ListResourceUsingListerFunc(f, resourceapi.SchemeGroupVersion.WithResource("resourceclaims"))
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
	return resourceapi.SchemeGroupVersion.WithResource("resourceclaims").GroupResource()
}

// Handles returns true if the evaluator should handle the specified operation.
func (p *claimEvaluator) Handles(a admission.Attributes) bool {
	if a.GetSubresource() != "" {
		return false
	}
	op := a.GetOperation()
	return admission.Create == op || admission.Update == op
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
		if item == ClaimObjectCountName /* object count quota fields */ ||
			strings.HasSuffix(string(item), corev1.ResourceClaimsPerClass /* by device class */) {
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
	result[ClaimObjectCountName] = *(resource.NewQuantity(1, resource.DecimalSI))
	for _, request := range claim.Spec.Devices.Requests {
		if len(request.FirstAvailable) > 0 {
			// If there are subrequests, we want to use the worst case per device class
			// to quota. So for each device class, we need to find the max number of
			// devices that might be allocated.
			maxQuantityByDeviceClassClaim := make(map[corev1.ResourceName]resource.Quantity)
			for _, subrequest := range request.FirstAvailable {
				deviceClassClaim := V1ResourceByDeviceClass(subrequest.DeviceClassName)
				var numDevices int64
				switch subrequest.AllocationMode {
				case resourceapi.DeviceAllocationModeExactCount:
					numDevices = subrequest.Count
				case resourceapi.DeviceAllocationModeAll:
					// Worst case...
					numDevices = resourceapi.AllocationResultsMaxSize
				default:
					// Could happen after a downgrade. Unknown modes
					// don't count towards the quota and users shouldn't
					// expect that when downgrading.
				}

				q := resource.NewQuantity(numDevices, resource.DecimalSI)
				if q.Cmp(maxQuantityByDeviceClassClaim[deviceClassClaim]) > 0 {
					maxQuantityByDeviceClassClaim[deviceClassClaim] = *q
				}
			}
			for deviceClassClaim, q := range maxQuantityByDeviceClassClaim {
				quantity := result[deviceClassClaim]
				quantity.Add(q)
				result[deviceClassClaim] = quantity
			}
			continue
		}
		deviceClassClaim := V1ResourceByDeviceClass(request.DeviceClassName)
		var numDevices int64
		switch request.AllocationMode {
		case resourceapi.DeviceAllocationModeExactCount:
			numDevices = request.Count
		case resourceapi.DeviceAllocationModeAll:
			// Worst case...
			numDevices = resourceapi.AllocationResultsMaxSize
		default:
			// Could happen after a downgrade. Unknown modes
			// don't count towards the quota and users shouldn't
			// expect that when downgrading.
		}
		quantity := result[deviceClassClaim]
		quantity.Add(*(resource.NewQuantity(numDevices, resource.DecimalSI)))
		result[deviceClassClaim] = quantity
	}

	return result, nil
}

// UsageStats calculates aggregate usage for the object.
func (p *claimEvaluator) UsageStats(options quota.UsageStatsOptions) (quota.UsageStats, error) {
	return generic.CalculateUsageStats(options, p.listFuncByNamespace, generic.MatchesNoScopeFunc, p.Usage)
}

// ensure we implement required interface
var _ quota.Evaluator = &claimEvaluator{}

func toExternalResourceClaimOrError(obj runtime.Object) (*resourceapi.ResourceClaim, error) {
	claim := &resourceapi.ResourceClaim{}
	switch t := obj.(type) {
	case *resourceapi.ResourceClaim:
		claim = t
	case *resourceinternal.ResourceClaim:
		if err := resourceversioned.Convert_resource_ResourceClaim_To_v1beta1_ResourceClaim(t, claim, nil); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("expect %T or %T, got %v", &resourceapi.ResourceClaim{}, &resourceinternal.ResourceClaim{}, t)
	}
	return claim, nil
}
