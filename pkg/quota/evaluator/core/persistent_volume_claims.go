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

// NewPersistentVolumeClaimEvaluator returns an evaluator that can evaluate persistent volume claims
func NewPersistentVolumeClaimEvaluator(kubeClient clientset.Interface) quota.Evaluator {
	allResources := []api.ResourceName{api.ResourcePersistentVolumeClaims, api.ResourceRequestsStorage}
	return &generic.GenericEvaluator{
		Name:              "Evaluator.PersistentVolumeClaim",
		InternalGroupKind: api.Kind("PersistentVolumeClaim"),
		InternalOperationResources: map[admission.Operation][]api.ResourceName{
			admission.Create: allResources,
		},
		MatchedResourceNames: allResources,
		MatchesScopeFunc:     generic.MatchesNoScopeFunc,
		ConstraintsFunc:      PersistentVolumeClaimConstraintsFunc,
		UsageFunc:            PersistentVolumeClaimUsageFunc,
		ListFuncByNamespace: func(namespace string, options api.ListOptions) (runtime.Object, error) {
			return kubeClient.Core().PersistentVolumeClaims(namespace).List(options)
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
	}
	return result
}

// PersistentVolumeClaimConstraintsFunc verifies that all required resources are present on the claim
// In addition, it validates that the resources are valid (i.e. requests < limits)
func PersistentVolumeClaimConstraintsFunc(required []api.ResourceName, object runtime.Object) error {
	pvc, ok := object.(*api.PersistentVolumeClaim)
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
