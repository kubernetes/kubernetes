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
	"slices"
	"strings"

	corev1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	quota "k8s.io/apiserver/pkg/quota/v1"
	"k8s.io/apiserver/pkg/quota/v1/generic"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	corev1listers "k8s.io/client-go/listers/core/v1"
	resourcehelper "k8s.io/component-helpers/resource"
	"k8s.io/dynamic-resource-allocation/deviceclass/cache"
	resourceinternal "k8s.io/kubernetes/pkg/apis/resource"
	resourceversioned "k8s.io/kubernetes/pkg/apis/resource/v1"
	"k8s.io/kubernetes/pkg/features"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// The name used for object count quota. This evaluator takes over counting
// those because of it's GroupResource, so it has to implement this
// count.
var ClaimObjectCountName = generic.ObjectCountQuotaResourceNameFor(resourceapi.SchemeGroupVersion.WithResource("resourceclaims").GroupResource())

// V1ResourceByDeviceClass returns a quota resource name by device class.
// gpuclass -> gpuclass.deviceclass.resource.k8s.io/devices
func V1ResourceByDeviceClass(className string) corev1.ResourceName {
	return corev1.ResourceName(className + corev1.ResourceClaimsPerClass)
}

// extendedResourceToQuotaRequestResource returns a quota request extended resource name.
// example.com/gpu -> requests.example.com/gpu
func extendedResourceToQuotaRequestResource(extendedResourceName string) corev1.ResourceName {
	return corev1.ResourceName(corev1.DefaultResourceRequestsPrefix + extendedResourceName)
}

// deviceClassToQuotaRequestResource returns a quota request implicit extended resource name.
// gpuclass -> requests.deviceclass.resource.kubernetes.io/gpuclass
func deviceClassToQuotaRequestResource(className string) corev1.ResourceName {
	return corev1.ResourceName(corev1.ResourceImplicitExtendedClaimsPerClass + className)
}

// NewResourceClaimEvaluator returns an evaluator that can evaluate resource claims
func NewResourceClaimEvaluator(f quota.ListerForResourceFunc, m *cache.DeviceClassMapping, podsGetter corev1listers.PodLister) quota.Evaluator {
	listFuncByNamespace := generic.ListResourceUsingListerFunc(f, resourceapi.SchemeGroupVersion.WithResource("resourceclaims"))
	claimEvaluator := &claimEvaluator{listFuncByNamespace: listFuncByNamespace, deviceClassMapping: m, podsGetter: podsGetter}
	return claimEvaluator
}

// claimEvaluator knows how to evaluate quota usage for resource claims
type claimEvaluator struct {
	// listFuncByNamespace knows how to list resource claims
	listFuncByNamespace generic.ListFuncByNamespace
	// a global cache of device class and extended resource mapping
	deviceClassMapping *cache.DeviceClassMapping
	// podsGetter is used to get pods
	podsGetter corev1listers.PodLister
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
			continue
		}
		if utilfeature.DefaultFeatureGate.Enabled(features.DRAExtendedResource) {
			if strings.HasPrefix(string(item), corev1.ResourceImplicitExtendedClaimsPerClass /* by implicit extended resource name */) ||
				isExtendedResourceNameForQuota(item) /* by extended resource name */ {
				result = append(result, item)
			}
		}
	}
	return result
}

func (p *claimEvaluator) extendedResourceQuota(dcName string) (corev1.ResourceName, bool) {
	if name, ok := p.deviceClassMapping.Get(dcName); ok {
		return extendedResourceToQuotaRequestResource(name), true
	}
	return "", false
}

func isPodRequest(name corev1.ResourceName, quantity resource.Quantity, reqs []requestQuantity) (int, bool) {
	for i, rq := range reqs {
		if rq.name == name && rq.quantity.Cmp(quantity) == 0 {
			return i, true
		}
	}
	return 0, false
}

func (p *claimEvaluator) setResourceQuantity(resourceMap map[corev1.ResourceName]resource.Quantity, quantity resource.Quantity, deviceClassName string, isExtendedResourceClaim bool, reqs []requestQuantity) []requestQuantity {
	deviceClassClaim := V1ResourceByDeviceClass(deviceClassName)
	q := resourceMap[deviceClassClaim]
	q.Add(quantity)
	resourceMap[deviceClassClaim] = q
	if !utilfeature.DefaultFeatureGate.Enabled(features.DRAExtendedResource) {
		return reqs
	}
	implicitExtendedResourceClaim := deviceClassToQuotaRequestResource(deviceClassName)
	implicitIndex := 0
	isImplicitExtendedResourceRequest := false
	if isExtendedResourceClaim {
		implicitIndex, isImplicitExtendedResourceRequest = isPodRequest(implicitExtendedResourceClaim, quantity, reqs)
	}
	extendedResourceClaim, ok := p.extendedResourceQuota(deviceClassName)
	explictIndex := 0
	isExplicitExtendedResourceRequest := false
	if ok && !isImplicitExtendedResourceRequest && isExtendedResourceClaim {
		explictIndex, isExplicitExtendedResourceRequest = isPodRequest(extendedResourceClaim, quantity, reqs)
	}
	if !isExtendedResourceClaim || !isImplicitExtendedResourceRequest || isExplicitExtendedResourceRequest {
		q := resourceMap[implicitExtendedResourceClaim]
		q.Add(quantity)
		resourceMap[implicitExtendedResourceClaim] = q
	}
	if ok {
		if !isExtendedResourceClaim || isImplicitExtendedResourceRequest || !isExplicitExtendedResourceRequest {
			q := resourceMap[extendedResourceClaim]
			q.Add(quantity)
			resourceMap[extendedResourceClaim] = q
		}
	}
	if isImplicitExtendedResourceRequest {
		reqs = append(reqs[:implicitIndex], reqs[implicitIndex+1:]...)
		return reqs
	}
	if isExplicitExtendedResourceRequest {
		reqs = append(reqs[:explictIndex], reqs[explictIndex+1:]...)
		return reqs
	}
	return reqs
}

type requestQuantity struct {
	name     corev1.ResourceName
	quantity resource.Quantity
}

func (p *claimEvaluator) verifyOwner(claim *resourceapi.ResourceClaim) ([]requestQuantity, bool) {
	controllerRef := metav1.GetControllerOf(claim)
	if controllerRef == nil {
		return nil, false
	}
	if controllerRef.Kind != "Pod" || controllerRef.APIVersion != "v1" {
		return nil, false
	}
	if p.podsGetter == nil {
		return nil, false
	}
	pod, err := p.podsGetter.Pods(claim.Namespace).Get(controllerRef.Name)
	if err != nil {
		return nil, false
	}
	if controllerRef.UID != pod.UID {
		return nil, false
	}
	if pod.Status.ExtendedResourceClaimStatus != nil && pod.Status.ExtendedResourceClaimStatus.ResourceClaimName != claim.Name {
		return nil, false
	}
	reqs := make([]requestQuantity, 0)
	sumResourceContainer := make(map[corev1.ResourceName]resource.Quantity)
	containers := slices.Clone(pod.Spec.InitContainers)
	containers = append(containers, pod.Spec.Containers...)
	for i, c := range containers {
		for r, q := range c.Resources.Requests {
			if schedutil.IsDRAExtendedResourceName(r) {
				isInit := i < len(pod.Spec.InitContainers)
				isSidecar := c.RestartPolicy != nil && *c.RestartPolicy == corev1.ContainerRestartPolicyAlways
				if isInit && !isSidecar {
					var rq resource.Quantity
					sumResourceContainer[r] = rq
					continue
				}
				reqs = append(reqs, requestQuantity{name: extendedResourceToQuotaRequestResource(string(r)), quantity: q})
				sumQ, ok := sumResourceContainer[r]
				if !ok {
					sumResourceContainer[r] = q
					continue
				}
				sumQ.Add(q)
				sumResourceContainer[r] = sumQ
			}
		}
	}
	podReqs := resourcehelper.PodRequests(pod, resourcehelper.PodResourcesOptions{})
	for r, sumQ := range sumResourceContainer {
		for r1, maxQ := range podReqs {
			if r == r1 {
				if maxQ.Cmp(sumQ) > 0 {
					maxQ.Sub(sumQ)
					reqs = append(reqs, requestQuantity{name: extendedResourceToQuotaRequestResource(string(r)), quantity: maxQ})
				}
			}
		}
	}
	return reqs, true
}

// Usage knows how to measure usage associated with item.
func (p *claimEvaluator) Usage(item runtime.Object) (corev1.ResourceList, error) {
	result := corev1.ResourceList{}
	claim, err := toExternalResourceClaimOrError(item)
	if err != nil {
		return result, err
	}

	isExtendedResourceClaim := false
	var reqs []requestQuantity

	if claim.Annotations[resourceapi.ExtendedResourceClaimAnnotation] == "true" {
		reqs, isExtendedResourceClaim = p.verifyOwner(claim)
	}
	// charge for claim
	result[ClaimObjectCountName] = *(resource.NewQuantity(1, resource.DecimalSI))
	for _, request := range claim.Spec.Devices.Requests {
		switch {
		case len(request.FirstAvailable) > 0:
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
					reqs = p.setResourceQuantity(maxQuantityByDeviceClassClaim, *q, subrequest.DeviceClassName, isExtendedResourceClaim, reqs)
				}
			}
			for deviceClassClaim, q := range maxQuantityByDeviceClassClaim {
				quantity := result[deviceClassClaim]
				quantity.Add(q)
				result[deviceClassClaim] = quantity
			}
			continue
		case request.Exactly != nil:
			var numDevices int64
			switch request.Exactly.AllocationMode {
			case resourceapi.DeviceAllocationModeExactCount:
				numDevices = request.Exactly.Count
			case resourceapi.DeviceAllocationModeAll:
				// Worst case...
				numDevices = resourceapi.AllocationResultsMaxSize
			default:
				// Could happen after a downgrade. Unknown modes
				// don't count towards the quota and users shouldn't
				// expect that when downgrading.
			}
			q := resource.NewQuantity(numDevices, resource.DecimalSI)
			reqs = p.setResourceQuantity(result, *q, request.Exactly.DeviceClassName, isExtendedResourceClaim, reqs)
		default:
			// Some unknown, future request type. Cannot do quota for it.
		}
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
		if err := resourceversioned.Convert_resource_ResourceClaim_To_v1_ResourceClaim(t, claim, nil); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("expect %T or %T, got %v", &resourceapi.ResourceClaim{}, &resourceinternal.ResourceClaim{}, t)
	}
	return claim, nil
}
