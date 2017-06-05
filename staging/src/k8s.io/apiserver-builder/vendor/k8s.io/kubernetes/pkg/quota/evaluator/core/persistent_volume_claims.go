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

package core

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/quota/generic"
)

// pvcResources are the set of static resources managed by quota associated with pvcs.
// for each resouce in this list, it may be refined dynamically based on storage class.
var pvcResources = []api.ResourceName{
	api.ResourcePersistentVolumeClaims,
	api.ResourceRequestsStorage,
}

// storageClassSuffix is the suffix to the qualified portion of storage class resource name.
// For example, if you want to quota storage by storage class, you would have a declaration
// that follows <storage-class>.storageclass.storage.k8s.io/<resource>.
// For example:
// * gold.storageclass.storage.k8s.io/: 500Gi
// * bronze.storageclass.storage.k8s.io/requests.storage: 500Gi
const storageClassSuffix string = ".storageclass.storage.k8s.io/"

// ResourceByStorageClass returns a quota resource name by storage class.
func ResourceByStorageClass(storageClass string, resourceName api.ResourceName) api.ResourceName {
	return api.ResourceName(string(storageClass + storageClassSuffix + string(resourceName)))
}

// V1ResourceByStorageClass returns a quota resource name by storage class.
func V1ResourceByStorageClass(storageClass string, resourceName v1.ResourceName) v1.ResourceName {
	return v1.ResourceName(string(storageClass + storageClassSuffix + string(resourceName)))
}

// listPersistentVolumeClaimsByNamespaceFuncUsingClient returns a pvc listing function based on the provided client.
func listPersistentVolumeClaimsByNamespaceFuncUsingClient(kubeClient clientset.Interface) generic.ListFuncByNamespace {
	// TODO: ideally, we could pass dynamic client pool down into this code, and have one way of doing this.
	// unfortunately, dynamic client works with Unstructured objects, and when we calculate Usage, we require
	// structured objects.
	return func(namespace string, options metav1.ListOptions) ([]runtime.Object, error) {
		itemList, err := kubeClient.Core().PersistentVolumeClaims(namespace).List(options)
		if err != nil {
			return nil, err
		}
		results := make([]runtime.Object, 0, len(itemList.Items))
		for i := range itemList.Items {
			results = append(results, &itemList.Items[i])
		}
		return results, nil
	}
}

// NewPersistentVolumeClaimEvaluator returns an evaluator that can evaluate persistent volume claims
// if the specified shared informer factory is not nil, evaluator may use it to support listing functions.
func NewPersistentVolumeClaimEvaluator(kubeClient clientset.Interface, f informers.SharedInformerFactory) quota.Evaluator {
	listFuncByNamespace := listPersistentVolumeClaimsByNamespaceFuncUsingClient(kubeClient)
	if f != nil {
		listFuncByNamespace = generic.ListResourceUsingInformerFunc(f, v1.SchemeGroupVersion.WithResource("persistentvolumeclaims"))
	}
	return &pvcEvaluator{
		listFuncByNamespace: listFuncByNamespace,
	}
}

// pvcEvaluator knows how to evaluate quota usage for persistent volume claims
type pvcEvaluator struct {
	// listFuncByNamespace knows how to list pvc claims
	listFuncByNamespace generic.ListFuncByNamespace
}

// Constraints verifies that all required resources are present on the item.
func (p *pvcEvaluator) Constraints(required []api.ResourceName, item runtime.Object) error {
	pvc, ok := item.(*api.PersistentVolumeClaim)
	if !ok {
		return fmt.Errorf("unexpected input object %v", item)
	}

	// these are the items that we will be handling based on the objects actual storage-class
	pvcRequiredSet := append([]api.ResourceName{}, pvcResources...)
	if storageClassRef := api.GetPersistentVolumeClaimClass(pvc); len(storageClassRef) > 0 {
		pvcRequiredSet = append(pvcRequiredSet, ResourceByStorageClass(storageClassRef, api.ResourcePersistentVolumeClaims))
		pvcRequiredSet = append(pvcRequiredSet, ResourceByStorageClass(storageClassRef, api.ResourceRequestsStorage))
	}

	// in effect, this will remove things from the required set that are not tied to this pvcs storage class
	// for example, if a quota has bronze and gold storage class items defined, we should not error a bronze pvc for not being gold.
	// but we should error a bronze pvc if it doesn't make a storage request size...
	requiredResources := quota.Intersection(required, pvcRequiredSet)
	requiredSet := quota.ToSet(requiredResources)

	// usage for this pvc will only include global pvc items + this storage class specific items
	pvcUsage, err := p.Usage(item)
	if err != nil {
		return err
	}

	// determine what required resources were not tracked by usage.
	missingSet := sets.NewString()
	pvcSet := quota.ToSet(quota.ResourceNames(pvcUsage))
	if diff := requiredSet.Difference(pvcSet); len(diff) > 0 {
		missingSet.Insert(diff.List()...)
	}
	if len(missingSet) == 0 {
		return nil
	}
	return fmt.Errorf("must specify %s", strings.Join(missingSet.List(), ","))
}

// GroupKind that this evaluator tracks
func (p *pvcEvaluator) GroupKind() schema.GroupKind {
	return api.Kind("PersistentVolumeClaim")
}

// Handles returns true if the evaluator should handle the specified operation.
func (p *pvcEvaluator) Handles(operation admission.Operation) bool {
	return admission.Create == operation
}

// Matches returns true if the evaluator matches the specified quota with the provided input item
func (p *pvcEvaluator) Matches(resourceQuota *api.ResourceQuota, item runtime.Object) (bool, error) {
	return generic.Matches(resourceQuota, item, p.MatchingResources, generic.MatchesNoScopeFunc)
}

// MatchingResources takes the input specified list of resources and returns the set of resources it matches.
func (p *pvcEvaluator) MatchingResources(items []api.ResourceName) []api.ResourceName {
	result := []api.ResourceName{}
	for _, item := range items {
		if quota.Contains(pvcResources, item) {
			result = append(result, item)
			continue
		}
		// match pvc resources scoped by storage class (<storage-class-name>.storage-class.kubernetes.io/<resource>)
		for _, resource := range pvcResources {
			byStorageClass := storageClassSuffix + string(resource)
			if strings.HasSuffix(string(item), byStorageClass) {
				result = append(result, item)
				break
			}
		}
	}
	return result
}

// Usage knows how to measure usage associated with item.
func (p *pvcEvaluator) Usage(item runtime.Object) (api.ResourceList, error) {
	result := api.ResourceList{}
	pvc, err := toInternalPersistentVolumeClaimOrError(item)
	if err != nil {
		return result, err
	}
	storageClassRef := api.GetPersistentVolumeClaimClass(pvc)

	// charge for claim
	result[api.ResourcePersistentVolumeClaims] = resource.MustParse("1")
	if len(storageClassRef) > 0 {
		storageClassClaim := api.ResourceName(storageClassRef + storageClassSuffix + string(api.ResourcePersistentVolumeClaims))
		result[storageClassClaim] = resource.MustParse("1")
	}

	// charge for storage
	if request, found := pvc.Spec.Resources.Requests[api.ResourceStorage]; found {
		result[api.ResourceRequestsStorage] = request
		// charge usage to the storage class (if present)
		if len(storageClassRef) > 0 {
			storageClassStorage := api.ResourceName(storageClassRef + storageClassSuffix + string(api.ResourceRequestsStorage))
			result[storageClassStorage] = request
		}
	}
	return result, nil
}

// UsageStats calculates aggregate usage for the object.
func (p *pvcEvaluator) UsageStats(options quota.UsageStatsOptions) (quota.UsageStats, error) {
	return generic.CalculateUsageStats(options, p.listFuncByNamespace, generic.MatchesNoScopeFunc, p.Usage)
}

// ensure we implement required interface
var _ quota.Evaluator = &pvcEvaluator{}

func toInternalPersistentVolumeClaimOrError(obj runtime.Object) (*api.PersistentVolumeClaim, error) {
	pvc := &api.PersistentVolumeClaim{}
	switch t := obj.(type) {
	case *v1.PersistentVolumeClaim:
		if err := v1.Convert_v1_PersistentVolumeClaim_To_api_PersistentVolumeClaim(t, pvc, nil); err != nil {
			return nil, err
		}
	case *api.PersistentVolumeClaim:
		pvc = t
	default:
		return nil, fmt.Errorf("expect *api.PersistentVolumeClaim or *v1.PersistentVolumeClaim, got %v", t)
	}
	return pvc, nil
}
