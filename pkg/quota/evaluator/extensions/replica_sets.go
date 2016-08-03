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

package extensions

import (
	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	extensionsapi "k8s.io/kubernetes/pkg/apis/extensions"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/quota/generic"
	"k8s.io/kubernetes/pkg/runtime"
)

// NewReplicaSetEvaluator returns an evaluator that can evaluate replica sets
func NewReplicaSetEvaluator(kubeClient clientset.Interface) quota.Evaluator {
	allResources := []api.ResourceName{extensionsapi.ResourceReplicaSets}
	return &generic.GenericEvaluator{
		Name:              "Evaluator.ReplicaSet",
		InternalGroupKind: extensionsapi.Kind("ReplicaSet"),
		InternalOperationResources: map[admission.Operation][]api.ResourceName{
			admission.Create: allResources,
		},
		MatchedResourceNames: allResources,
		MatchesScopeFunc:     generic.MatchesNoScopeFunc,
		ConstraintsFunc:      generic.ObjectCountConstraintsFunc(extensionsapi.ResourceReplicaSets),
		UsageFunc:            generic.ObjectCountUsageFunc(extensionsapi.ResourceReplicaSets),
		ListFuncByNamespace: func(namespace string, options api.ListOptions) (runtime.Object, error) {
			return kubeClient.Extensions().ReplicaSets(namespace).List(options)
		},
	}
}
