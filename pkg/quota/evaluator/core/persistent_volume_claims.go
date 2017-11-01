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

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/initialization"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/helper"
	k8s_api_v1 "k8s.io/kubernetes/pkg/api/v1"
	k8sfeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubeapiserver/admission/util"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/quota/generic"
)

// the name used for object count quota
var pvcObjectCountName = generic.ObjectCountQuotaResourceNameFor(v1.SchemeGroupVersion.WithResource("persistentvolumeclaims").GroupResource())

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

// NewPersistentVolumeClaimEvaluator returns an evaluator that can evaluate persistent volume claims
func NewPersistentVolumeClaimEvaluator(f quota.ListerForResourceFunc) quota.Evaluator {
	listFuncByNamespace := generic.ListResourceUsingListerFunc(f, v1.SchemeGroupVersion.WithResource("persistentvolumeclaims"))
	pvcEvaluator := &pvcEvaluator{listFuncByNamespace: listFuncByNamespace}
	return pvcEvaluator
}

// pvcEvaluator knows how to evaluate quota usage for persistent volume claims
type pvcEvaluator struct {
	// listFuncByNamespace knows how to list pvc claims
	listFuncByNamespace generic.ListFuncByNamespace
}

// Constraints verifies that all required resources are present on the item.
func (p *pvcEvaluator) Constraints(required []api.ResourceName, item runtime.Object) error {
	// no-op for persistent volume claims
	return nil
}

// GroupResource that this evaluator tracks
func (p *pvcEvaluator) GroupResource() schema.GroupResource {
	return v1.SchemeGroupVersion.WithResource("persistentvolumeclaims").GroupResource()
}

// Handles returns true if the evaluator should handle the specified operation.
func (p *pvcEvaluator) Handles(a admission.Attributes) bool {
	op := a.GetOperation()
	if op == admission.Create {
		return true
	}
	if op == admission.Update && utilfeature.DefaultFeatureGate.Enabled(k8sfeatures.ExpandPersistentVolumes) {
		initialized, err := initialization.IsObjectInitialized(a.GetObject())
		if err != nil {
			// fail closed, will try to give an evaluation.
			utilruntime.HandleError(err)
			return true
		}
		// only handle the update if the object is initialized after the update.
		return initialized
	}
	// TODO: when the ExpandPersistentVolumes feature gate is removed, remove
	// the initializationCompletion check as well, because it will become a
	// subset of the "initialized" condition.
	initializationCompletion, err := util.IsInitializationCompletion(a)
	if err != nil {
		// fail closed, will try to give an evaluation.
		utilruntime.HandleError(err)
		return true
	}
	return initializationCompletion
}

// Matches returns true if the evaluator matches the specified quota with the provided input item
func (p *pvcEvaluator) Matches(resourceQuota *api.ResourceQuota, item runtime.Object) (bool, error) {
	return generic.Matches(resourceQuota, item, p.MatchingResources, generic.MatchesNoScopeFunc)
}

// MatchingResources takes the input specified list of resources and returns the set of resources it matches.
func (p *pvcEvaluator) MatchingResources(items []api.ResourceName) []api.ResourceName {
	result := []api.ResourceName{}
	for _, item := range items {
		// match object count quota fields
		if quota.Contains([]api.ResourceName{pvcObjectCountName}, item) {
			result = append(result, item)
			continue
		}
		// match pvc resources
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

	// charge for claim
	result[api.ResourcePersistentVolumeClaims] = *(resource.NewQuantity(1, resource.DecimalSI))
	result[pvcObjectCountName] = *(resource.NewQuantity(1, resource.DecimalSI))
	if utilfeature.DefaultFeatureGate.Enabled(features.Initializers) {
		if !initialization.IsInitialized(pvc.Initializers) {
			// Only charge pvc count for uninitialized pvc.
			return result, nil
		}
	}
	storageClassRef := helper.GetPersistentVolumeClaimClass(pvc)
	if len(storageClassRef) > 0 {
		storageClassClaim := api.ResourceName(storageClassRef + storageClassSuffix + string(api.ResourcePersistentVolumeClaims))
		result[storageClassClaim] = *(resource.NewQuantity(1, resource.DecimalSI))
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
		if err := k8s_api_v1.Convert_v1_PersistentVolumeClaim_To_api_PersistentVolumeClaim(t, pvc, nil); err != nil {
			return nil, err
		}
	case *api.PersistentVolumeClaim:
		pvc = t
	default:
		return nil, fmt.Errorf("expect *api.PersistentVolumeClaim or *v1.PersistentVolumeClaim, got %v", t)
	}
	return pvc, nil
}
