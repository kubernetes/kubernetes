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
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission"
	quota "k8s.io/apiserver/pkg/quota/v1"
	"k8s.io/apiserver/pkg/quota/v1/generic"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	v1 "k8s.io/client-go/informers/resource/v1"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/dynamic-resource-allocation/deviceclass/extendedresourcecache"
	resourceinternal "k8s.io/kubernetes/pkg/apis/resource"
	resourceversioned "k8s.io/kubernetes/pkg/apis/resource/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/clock"
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

// NewResourceClaimEvaluator returns an evaluator that can evaluate resource claims
func NewResourceClaimEvaluator(f quota.ListerForResourceFunc, m *extendedresourcecache.ExtendedResourceCache, podsGetter corev1listers.PodLister, claimGetter resourceClaimPodOwnerGetter) quota.Evaluator {
	listFuncByNamespace := generic.ListResourceUsingListerFunc(f, resourceapi.SchemeGroupVersion.WithResource("resourceclaims"))
	claimEvaluator := &claimEvaluator{listFuncByNamespace: listFuncByNamespace, deviceClassMapping: m, podsGetter: podsGetter, claimGetter: claimGetter}
	return claimEvaluator
}

// claimEvaluator knows how to evaluate quota usage for resource claims
type claimEvaluator struct {
	// listFuncByNamespace knows how to list resource claims
	listFuncByNamespace generic.ListFuncByNamespace
	// a global cache of device class and extended resource mapping
	deviceClassMapping *extendedresourcecache.ExtendedResourceCache
	// podsGetter is used to get pods
	podsGetter corev1listers.PodLister
	// claimGetter is used to get claims for extended resources by namespace and pod owner uid
	claimGetter resourceClaimPodOwnerGetter
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
			if strings.HasPrefix(string(item), corev1.ResourceImplicitExtendedClaimsPerClass /* by implicit extended resource name */) {
				className := string(item[len(corev1.ResourceImplicitExtendedClaimsPerClass):])
				if p.deviceClassMapping.GetExtendedResource(className) != "" {
					result = append(result, item)
					continue
				}
			}
			if isExtendedResourceNameForQuota(item) /* by extended resource name */ {
				resourceName := string(item[len(corev1.DefaultResourceRequestsPrefix):])
				if p.deviceClassMapping.GetDeviceClass(corev1.ResourceName(resourceName)) != nil {
					result = append(result, item)
				}
			}
		}
	}
	return result
}

func (p *claimEvaluator) addExtendedResourceQuota(resourceClaimUsage map[corev1.ResourceName]resource.Quantity, podUsage corev1.ResourceList) {
	extendedResourceUsage := make(map[corev1.ResourceName]resource.Quantity)
	for name, quantity := range resourceClaimUsage {
		// e.g. myclass
		deviceClassName, isDeviceClassUsage := strings.CutSuffix(string(name), corev1.ResourceClaimsPerClass)
		if !isDeviceClassUsage || len(deviceClassName) == 0 {
			continue
		}

		// requests.deviceclass.resource.kubernetes.io/myclass
		extendedResourceUsage[corev1.ResourceName(corev1.ResourceImplicitExtendedClaimsPerClass+deviceClassName)] = quantity

		// e.g. example.com/mygpu
		if extendedResourceName := p.deviceClassMapping.GetExtendedResource(deviceClassName); len(extendedResourceName) > 0 {
			// requests.example.com/gpu
			extendedResourceUsage[corev1.ResourceName(corev1.DefaultResourceRequestsPrefix+extendedResourceName)] = quantity
		}
	}

	for name, quantity := range extendedResourceUsage {
		// Subtract any amount already accounted for in the pod
		if podQuantity, found := podUsage[name]; found {
			quantity.Sub(podQuantity)
		}
		// Add any remaining amount to the resource claim resources
		if quantity.CmpInt64(0) > 0 {
			resourceClaimUsage[name] = quantity
		}
	}
}

// Verify extended resource claim owning pod exists, and the pod's ExtendedResourceClaimStatus points
// back to the claim if it's not nil, and returns the pod's quota usage. If any error is encountered, nil is returned.
func (p *claimEvaluator) getVerifiedPodUsage(claim *resourceapi.ResourceClaim) corev1.ResourceList {
	if claim.Annotations[resourceapi.ExtendedResourceClaimAnnotation] != "true" {
		return nil
	}
	controllerRef := metav1.GetControllerOfNoCopy(claim)
	if controllerRef == nil {
		return nil
	}
	if controllerRef.Kind != "Pod" || controllerRef.APIVersion != "v1" {
		return nil
	}
	if p.podsGetter == nil {
		return nil
	}
	pod, err := p.podsGetter.Pods(claim.Namespace).Get(controllerRef.Name)
	if err != nil {
		return nil
	}
	if controllerRef.UID != pod.UID {
		return nil
	}
	if pod.Status.ExtendedResourceClaimStatus != nil && pod.Status.ExtendedResourceClaimStatus.ResourceClaimName != claim.Name {
		return nil
	}
	// if the pod doesn't identify its extended resource claim, make sure we're the first or only extended resource claim for this pod
	if pod.Status.ExtendedResourceClaimStatus == nil {
		ownedClaims, err := p.claimGetter(claim.Namespace, pod.UID)
		if err != nil {
			return nil
		}
		for _, ownedClaim := range ownedClaims {
			switch ownedClaim.CreationTimestamp.Time.Compare(claim.CreationTimestamp.Time) {
			case -1:
				// There's another owned claim created earlier than this one.
				// Don't exempt this one based on pod usage.
				return nil
			case 0:
				if ownedClaim.Name < claim.Name {
					// There's another owned claim created at the same time as this one with an earlier name.
					// Don't exempt this one based on pod usage.
					return nil
				}
			case 1:
				continue
			}
		}
	}

	quotaReqs, err := PodUsageFunc(pod, clock.RealClock{})
	if err != nil {
		return nil
	}
	return quotaReqs
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
					maxQuantityByDeviceClassClaim[deviceClassClaim] = *q
				}
			}
			for deviceClassClaim, q := range maxQuantityByDeviceClassClaim {
				quantity := result[deviceClassClaim]
				quantity.Add(q)
				result[deviceClassClaim] = quantity
			}
			continue
		case request.Exactly != nil:
			deviceClassClaim := V1ResourceByDeviceClass(request.Exactly.DeviceClassName)
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
			quantity := result[deviceClassClaim]
			quantity.Add(*(resource.NewQuantity(numDevices, resource.DecimalSI)))
			result[deviceClassClaim] = quantity
		default:
			// Some unknown, future request type. Cannot do quota for it.
		}
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.DRAExtendedResource) {
		p.addExtendedResourceQuota(result, p.getVerifiedPodUsage(claim))
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

const extendedResourceClaimPodOwnerIndexName = "extendedResourceClaimPodUIDOwner"

func resourceClaimPodOwnerKey(namespace string, podUID types.UID) string {
	return namespace + "/" + string(podUID)
}

type resourceClaimPodOwnerGetter func(namespace string, podUID types.UID) ([]*resourceapi.ResourceClaim, error)

func makeResourceClaimPodOwnerGetter(resourceClaimInformer v1.ResourceClaimInformer) (resourceClaimPodOwnerGetter, error) {
	indexers := cache.Indexers{extendedResourceClaimPodOwnerIndexName: extendedResourceClaimPodUIDOwnerIndex}
	if err := resourceClaimInformer.Informer().AddIndexers(indexers); err != nil {
		_, exists := resourceClaimInformer.Informer().GetIndexer().GetIndexers()[extendedResourceClaimPodOwnerIndexName]
		if !exists {
			return nil, fmt.Errorf("failed to add resource claim pod owner indexer: %w", err)
		}
	}
	indexer := resourceClaimInformer.Informer().GetIndexer()
	return func(namespace string, podUID types.UID) ([]*resourceapi.ResourceClaim, error) {
		objects, err := indexer.ByIndex(extendedResourceClaimPodOwnerIndexName, resourceClaimPodOwnerKey(namespace, podUID))
		if err != nil {
			return nil, err
		}
		claims := make([]*resourceapi.ResourceClaim, 0, len(objects))
		for _, obj := range objects {
			if claim, ok := obj.(*resourceapi.ResourceClaim); ok {
				claims = append(claims, claim)
			} else {
				return nil, fmt.Errorf("failed to get resource claim from indexer")
			}
		}
		return claims, nil
	}, nil
}
func extendedResourceClaimPodUIDOwnerIndex(obj interface{}) ([]string, error) {
	claim, ok := obj.(*resourceapi.ResourceClaim)
	if !ok {
		return nil, nil
	}
	if claim.Annotations[resourceapi.ExtendedResourceClaimAnnotation] != "true" {
		return nil, nil
	}
	controllerRef := metav1.GetControllerOfNoCopy(claim)
	if controllerRef == nil {
		return nil, nil
	}
	if controllerRef.Kind != "Pod" || controllerRef.APIVersion != "v1" {
		return nil, nil
	}
	return []string{resourceClaimPodOwnerKey(claim.Namespace, controllerRef.UID)}, nil
}
