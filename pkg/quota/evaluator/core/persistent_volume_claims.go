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
	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/quota/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
)

// NewPersistentVolumeClaimEvaluator returns an evaluator that can evaluate persistent volume claims
func NewPersistentVolumeClaimEvaluator(kubeClient clientset.Interface) quota.Evaluator {
	allResources := []v1.ResourceName{v1.ResourcePersistentVolumeClaims, v1.ResourceRequestsStorage}
	return &generic.GenericEvaluator{
		Name:              "Evaluator.PersistentVolumeClaim",
		InternalGroupKind: api.Kind("PersistentVolumeClaim"),
		InternalOperationResources: map[admission.Operation][]v1.ResourceName{
			admission.Create: allResources,
		},
		MatchedResourceNames: allResources,
		MatchesScopeFunc:     generic.MatchesNoScopeFunc,
		ConstraintsFunc:      PersistentVolumeClaimConstraintsFunc,
		UsageFunc:            PersistentVolumeClaimUsageFunc,
		ListFuncByNamespace: func(namespace string, options v1.ListOptions) ([]runtime.Object, error) {
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
func PersistentVolumeClaimUsageFunc(object runtime.Object) v1.ResourceList {
	pvc, ok := object.(*v1.PersistentVolumeClaim)
	if !ok {
		return v1.ResourceList{}
	}
	result := v1.ResourceList{}
	result[v1.ResourcePersistentVolumeClaims] = resource.MustParse("1")
	if request, found := pvc.Spec.Resources.Requests[v1.ResourceStorage]; found {
		result[v1.ResourceRequestsStorage] = request
	}
	return result
}

// PersistentVolumeClaimConstraintsFunc verifies that all required resources are present on the claim
// In addition, it validates that the resources are valid (i.e. requests < limits)
func PersistentVolumeClaimConstraintsFunc(required []v1.ResourceName, object runtime.Object) error {
	pvc, ok := object.(*v1.PersistentVolumeClaim)
	if !ok {
		return fmt.Errorf("unexpected input object %v", object)
	}

	requiredSet := quota.ToSet(required)
	missingSet := sets.NewString()
	pvcUsage := PersistentVolumeClaimUsageFunc(pvc)
	pvcSet := quota.ToSet(quota.ResourceNames(pvcUsage))
	if diff := requiredSet.Difference(pvcSet); len(diff) > 0 {
		missingSet.Insert(diff.List()...)
	}
	if len(missingSet) == 0 {
		return nil
	}
	return fmt.Errorf("must specify %s", strings.Join(missingSet.List(), ","))
}
