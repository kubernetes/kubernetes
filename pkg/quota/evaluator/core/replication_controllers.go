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
	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/quota/generic"
	"k8s.io/kubernetes/pkg/runtime"
)

// NewReplicationControllerEvaluator returns an evaluator that can evaluate replication controllers
func NewReplicationControllerEvaluator(kubeClient clientset.Interface) quota.Evaluator {
	allResources := []api.ResourceName{api.ResourceReplicationControllers}
	return &generic.GenericEvaluator{
		Name:              "Evaluator.ReplicationController",
		InternalGroupKind: api.Kind("ReplicationController"),
		InternalOperationResources: map[admission.Operation][]api.ResourceName{
			admission.Create: allResources,
		},
		MatchedResourceNames: allResources,
		MatchesScopeFunc:     generic.MatchesNoScopeFunc,
		ConstraintsFunc:      generic.ObjectCountConstraintsFunc(api.ResourceReplicationControllers),
		UsageFunc:            generic.ObjectCountUsageFunc(api.ResourceReplicationControllers),
		ListFuncByNamespace: func(namespace string, options v1.ListOptions) ([]runtime.Object, error) {
			itemList, err := kubeClient.Core().ReplicationControllers(namespace).List(options)
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
