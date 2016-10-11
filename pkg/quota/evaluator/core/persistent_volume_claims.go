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

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/quota/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
)

const storageClassSuffix string = ".storage-class.kubernetes.io/"

// NewPersistentVolumeClaimEvaluator returns an evaluator that can evaluate persistent volume claims
func NewPersistentVolumeClaimEvaluator(kubeClient clientset.Interface) quota.Evaluator {
	fixedResources := []api.ResourceName{api.ResourcePersistentVolumeClaims, api.ResourceRequestsStorage}
	usageFunc := PersistentVolumeClaimUsageFunc
	constraintsFunc := makePersistentVolumeClaimConstraintsFunc(usageFunc)
	return &generic.GenericEvaluator{
		Name:              "Evaluator.PersistentVolumeClaim",
		InternalGroupKind: api.Kind("PersistentVolumeClaim"),
		Operations:        []admission.Operation{admission.Create},
		MatchedResourceNamesFunc: func(items []api.ResourceName) (result []api.ResourceName) {
			for _, item := range items {
				if quota.Contains(fixedResources, item) {
					result = append(result, item)
					continue
				}
				// match pvc resources scoped by storage class (<storage-class-name>.storage-class.kubernetes.io/<resource>)
				for _, resource := range fixedResources {
					byStorageClass := storageClassSuffix + string(resource)
					if strings.HasSuffix(string(item), byStorageClass) {
						result = append(result, item)
						break
					}
				}
			}
			return result
		},
		MatchesScopeFunc: generic.MatchesNoScopeFunc,
		ConstraintsFunc:  constraintsFunc,
		UsageFunc:        usageFunc,
		ListFuncByNamespace: func(namespace string, options api.ListOptions) ([]runtime.Object, error) {
			itemList, err := kubeClient.Core().PersistentVolumeClaims(namespace).List(options)
			if err != nil {
				return nil, err
			}
			results := make([]runtime.Object, 0, len(itemList.Items))
			for i := range itemList.Items {
				results = append(results, &itemList.Items[i])
			}
			return results, nil
		},
	}
}

// PersistentVolumeClaimUsageFunc knows how to measure usage associated with persistent volume claims
func PersistentVolumeClaimUsageFunc(object runtime.Object) api.ResourceList {
	pvc, ok := object.(*api.PersistentVolumeClaim)
	if !ok {
		return api.ResourceList{}
	}
	result := api.ResourceList{}
	result[api.ResourcePersistentVolumeClaims] = resource.MustParse("1")
	if request, found := pvc.Spec.Resources.Requests[api.ResourceStorage]; found {
		result[api.ResourceRequestsStorage] = request
		// charge usage to the storage class (if present)
		// TODO: it would be great if there was a common utility to reference here so we can use it when promoting to beta...
		if storageClassRef, storageClassFound := pvc.Annotations["volume.beta.kubernetes.io/storage-class"]; storageClassFound {
			storageClassClaim := api.ResourceName(storageClassRef + storageClassSuffix + string(api.ResourcePersistentVolumeClaims))
			storageClassStorage := api.ResourceName(storageClassRef + storageClassSuffix + string(api.ResourceRequestsStorage))
			result[storageClassStorage] = request
			result[storageClassClaim] = resource.MustParse("1")
		}
	}
	return result
}

// makePersistentVolumeClaimConstraintsFunc returns a function that knows how to enforce constraints.
func makePersistentVolumeClaimConstraintsFunc(usageFunc generic.UsageFunc) generic.ConstraintsFunc {
	return func(required []api.ResourceName, object runtime.Object) error {
		pvc, ok := object.(*api.PersistentVolumeClaim)
		if !ok {
			return fmt.Errorf("unexpected input object %v", object)
		}

		// these are the items that we will be handling based on the objects actual storage-class
		pvcRequiredSet := []api.ResourceName{api.ResourceRequestsStorage, api.ResourcePersistentVolumeClaims}
		if storageClassRef, storageClassFound := pvc.Annotations["volume.beta.kubernetes.io/storage-class"]; storageClassFound {
			storageClassClaim := api.ResourceName(storageClassRef + storageClassSuffix + string(api.ResourcePersistentVolumeClaims))
			storageClassStorage := api.ResourceName(storageClassRef + storageClassSuffix + string(api.ResourceRequestsStorage))
			pvcRequiredSet = append(pvcRequiredSet, storageClassClaim)
			pvcRequiredSet = append(pvcRequiredSet, storageClassStorage)
		}

		// in effect, this will remove things from the required set that are not tied to this pvcs storage class
		// for example, if a quota has bronze and gold storage class items defined, we should not error a bronze pvc for not being gold.
		// but we should error a bronze pvc if it doesn't make a storage request size...
		requiredResources := quota.Intersection(required, pvcRequiredSet)
		requiredSet := quota.ToSet(requiredResources)

		// usage for this pvc will only include global pvc items + this storage class specific items
		pvcUsage := usageFunc(pvc)

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
}
