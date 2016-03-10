/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/quota/generic"
	"k8s.io/kubernetes/pkg/runtime"
)

// NewDaemonSetEvaluator returns an evaluator that can evaluate daemonsets
func NewDaemonSetEvaluator(kubeClient clientset.Interface) quota.Evaluator {
	allResources := []api.ResourceName{api.ResourceDaemonSets}
	return &generic.GenericEvaluator{
		Name:              "Evaluator.ResourceDaemonSet",
		InternalGroupKind: api.Kind("ResourceDaemonSet"),
		InternalOperationResources: map[admission.Operation][]api.ResourceName{
			admission.Create: allResources,
		},
		MatchedResourceNames: allResources,
		MatchesScopeFunc:     generic.MatchesNoScopeFunc,
		ConstraintsFunc:      generic.ObjectCountConstraintsFunc(api.ResourceDaemonSets),
		UsageFunc:            generic.ObjectCountUsageFunc(api.ResourceDaemonSets),
		ListFuncByNamespace: func(namespace string, options api.ListOptions) (runtime.Object, error) {
			return kubeClient.Extensions().DaemonSets(namespace).List(options)
		},
	}
}
