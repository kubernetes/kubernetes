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
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/quota/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/watch"
)

const storageClassSuffix string = ".storage-class.kubernetes.io/"

// NewPersistentVolumeClaimEvaluator returns an evaluator that can evaluate persistent volume claims
func NewPersistentVolumeClaimEvaluator(kubeClient clientset.Interface) quota.Evaluator {
	storageClassStore := cache.NewStore(cache.MetaNamespaceKeyFunc)
	reflector := cache.NewReflector(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return kubeClient.Storage().StorageClasses().List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return kubeClient.Storage().StorageClasses().Watch(options)
			},
		},
		&storage.StorageClass{},
		storageClassStore,
		0,
	)
	reflector.Run()

	matchesResourceNamesFunc := makeMatchedResourceNamesFunc(storageClassStore)
	usageFunc := makePersistentVolumeClaimUsageFunc(storageClassStore, matchesResourceNamesFunc)
	constraintsFunc := makePersistentVolumeClaimConstraintsFunc(usageFunc)
	allResources := []api.ResourceName{api.ResourcePersistentVolumeClaims, api.ResourceRequestsStorage}
	return &generic.GenericEvaluator{
		Name:              "Evaluator.PersistentVolumeClaim",
		InternalGroupKind: api.Kind("PersistentVolumeClaim"),
		InternalOperationResources: map[admission.Operation][]api.ResourceName{
			admission.Create: allResources,
		},
		MatchedResourceNames:     allResources,
		MatchedResourceNamesFunc: matchesResourceNamesFunc,
		MatchesScopeFunc:         generic.MatchesNoScopeFunc,
		ConstraintsFunc:          constraintsFunc,
		UsageFunc:                usageFunc,
		ListFuncByNamespace: func(namespace string, options api.ListOptions) (runtime.Object, error) {
			return kubeClient.Core().PersistentVolumeClaims(namespace).List(options)
		},
	}
}

// makeMatchedResourceNamesFunc returns a function that returns the list of resources matched
func makeMatchedResourceNamesFunc(storageClassStore cache.Store) generic.MatchesResourceNamesFunc {
	allResources := []api.ResourceName{api.ResourcePersistentVolumeClaims, api.ResourceRequestsStorage}
	return func() []api.ResourceName {
		resourceNames := []api.ResourceName{}
		for _, c := range storageClassStore.List() {
			if storageClass, ok := c.(*storage.StorageClass); ok {
				prefix := storageClass.Name
				for _, resource := range allResources {
					resourceName := api.ResourceName(prefix + storageClassSuffix + string(resource))
					resourceNames = append(resourceNames, resourceName)
				}
			}
		}
		fmt.Printf("quota: matchesfunc: %v\n", resourceNames)
		return resourceNames
	}
}

// makePersistentVolumeClaimUsageFunc returns a function that knows how to measure usage for pvcs
func makePersistentVolumeClaimUsageFunc(storageClassStore cache.Store, matchesResourceNameFunc generic.MatchesResourceNamesFunc) generic.UsageFunc {
	return func(object runtime.Object) api.ResourceList {
		pvc, ok := object.(*api.PersistentVolumeClaim)
		if !ok {
			return api.ResourceList{}
		}

		result := api.ResourceList{}
		matchedResources := matchesResourceNameFunc()
		for _, matchedResource := range matchedResources {
			result[matchedResource] = resource.MustParse("0")
		}

		result[api.ResourcePersistentVolumeClaims] = resource.MustParse("1")
		if request, found := pvc.Spec.Resources.Requests[api.ResourceStorage]; found {
			result[api.ResourceRequestsStorage] = request
			// charge usage to the storage class (if present)
			if storageClassRef, storageClassFound := pvc.Annotations["volume.beta.kubernetes.io/storage-class"]; storageClassFound {
				storageClassStorage := api.ResourceName(storageClassRef + storageClassSuffix + "requests.storage")
				storageClassClaim := api.ResourceName(storageClassRef + storageClassSuffix + "persistentvolumeclaims")
				result[storageClassStorage] = request
				result[storageClassClaim] = resource.MustParse("1")
			}
		}
		fmt.Printf("quota: usagefunc: %v\n", result)
		return result
	}
}

// makePersistentVolumeClaimConstraintsFunc returns a function that knows how to enforce constraints.
func makePersistentVolumeClaimConstraintsFunc(usageFunc generic.UsageFunc) generic.ConstraintsFunc {
	return func(required []api.ResourceName, object runtime.Object) error {
		pvc, ok := object.(*api.PersistentVolumeClaim)
		if !ok {
			return fmt.Errorf("unexpected input object %v", object)
		}

		requiredSet := quota.ToSet(required)
		missingSet := sets.NewString()
		pvcUsage := usageFunc(pvc)
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
