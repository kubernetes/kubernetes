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
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/kubelet/qos/util"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/quota/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
)

// NewPodEvaluator returns an evaluator that can evaluate pods
func NewPodEvaluator(kubeClient clientset.Interface) quota.Evaluator {
	computeResources := []api.ResourceName{
		api.ResourceCPU,
		api.ResourceMemory,
		api.ResourceRequestsCPU,
		api.ResourceRequestsMemory,
		api.ResourceLimitsCPU,
		api.ResourceLimitsMemory,
	}
	allResources := append(computeResources, api.ResourcePods)
	return &generic.GenericEvaluator{
		Name:              "Evaluator.Pod",
		InternalGroupKind: api.Kind("Pod"),
		InternalOperationResources: map[admission.Operation][]api.ResourceName{
			admission.Create: allResources,
			admission.Update: computeResources,
		},
		GetFuncByNamespace: func(namespace, name string) (runtime.Object, error) {
			return kubeClient.Core().Pods(namespace).Get(name)
		},
		ConstraintsFunc:      PodConstraintsFunc,
		MatchedResourceNames: allResources,
		MatchesScopeFunc:     PodMatchesScopeFunc,
		UsageFunc:            PodUsageFunc,
		ListFuncByNamespace: func(namespace string, options api.ListOptions) (runtime.Object, error) {
			return kubeClient.Core().Pods(namespace).List(options)
		},
	}
}

// PodConstraintsFunc verifies that all required resources are present on the pod
func PodConstraintsFunc(required []api.ResourceName, object runtime.Object) error {
	pod, ok := object.(*api.Pod)
	if !ok {
		return fmt.Errorf("Unexpected input object %v", object)
	}

	// TODO: fix this when we have pod level cgroups
	// since we do not yet pod level requests/limits, we need to ensure each
	// container makes an explict request or limit for a quota tracked resource
	requiredSet := quota.ToSet(required)
	missingSet := sets.NewString()
	for i := range pod.Spec.Containers {
		requests := pod.Spec.Containers[i].Resources.Requests
		limits := pod.Spec.Containers[i].Resources.Limits
		containerUsage := podUsageHelper(requests, limits)
		containerSet := quota.ToSet(quota.ResourceNames(containerUsage))
		if !containerSet.Equal(requiredSet) {
			difference := requiredSet.Difference(containerSet)
			missingSet.Insert(difference.List()...)
		}
	}
	if len(missingSet) == 0 {
		return nil
	}
	return fmt.Errorf("must specify %s", strings.Join(missingSet.List(), ","))
}

// podUsageHelper can summarize the pod quota usage based on requests and limits
func podUsageHelper(requests api.ResourceList, limits api.ResourceList) api.ResourceList {
	result := api.ResourceList{}
	result[api.ResourcePods] = resource.MustParse("1")
	if request, found := requests[api.ResourceCPU]; found {
		result[api.ResourceCPU] = request
		result[api.ResourceRequestsCPU] = request
	}
	if limit, found := limits[api.ResourceCPU]; found {
		result[api.ResourceLimitsCPU] = limit
	}
	if request, found := requests[api.ResourceMemory]; found {
		result[api.ResourceMemory] = request
		result[api.ResourceRequestsMemory] = request
	}
	if limit, found := limits[api.ResourceMemory]; found {
		result[api.ResourceLimitsMemory] = limit
	}
	return result
}

// PodUsageFunc knows how to measure usage associated with pods
func PodUsageFunc(object runtime.Object) api.ResourceList {
	pod, ok := object.(*api.Pod)
	if !ok {
		return api.ResourceList{}
	}

	// by convention, we do not quota pods that have reached an end-of-life state
	if !QuotaPod(pod) {
		return api.ResourceList{}
	}

	// TODO: fix this when we have pod level cgroups
	// when we have pod level cgroups, we can just read pod level requests/limits
	requests := api.ResourceList{}
	limits := api.ResourceList{}
	for i := range pod.Spec.Containers {
		requests = quota.Add(requests, pod.Spec.Containers[i].Resources.Requests)
		limits = quota.Add(limits, pod.Spec.Containers[i].Resources.Limits)
	}

	return podUsageHelper(requests, limits)
}

// PodMatchesScopeFunc is a function that knows how to evaluate if a pod matches a scope
func PodMatchesScopeFunc(scope api.ResourceQuotaScope, object runtime.Object) bool {
	pod, ok := object.(*api.Pod)
	if !ok {
		return false
	}
	switch scope {
	case api.ResourceQuotaScopeTerminating:
		return isTerminating(pod)
	case api.ResourceQuotaScopeNotTerminating:
		return !isTerminating(pod)
	case api.ResourceQuotaScopeBestEffort:
		return isBestEffort(pod)
	case api.ResourceQuotaScopeNotBestEffort:
		return !isBestEffort(pod)
	}
	return false
}

func isBestEffort(pod *api.Pod) bool {
	// TODO: when we have request/limits on a pod scope, we need to revisit this
	for _, container := range pod.Spec.Containers {
		qosPerResource := util.GetQoS(&container)
		for _, qos := range qosPerResource {
			if util.BestEffort == qos {
				return true
			}
		}
	}
	return false
}

func isTerminating(pod *api.Pod) bool {
	if pod.Spec.ActiveDeadlineSeconds != nil && *pod.Spec.ActiveDeadlineSeconds >= int64(0) {
		return true
	}
	return false
}

// QuotaPod returns true if the pod is eligible to track against a quota
// if it's not in a terminal state according to its phase.
func QuotaPod(pod *api.Pod) bool {
	// see GetPhase in kubelet.go for details on how it covers all restart policy conditions
	// https://github.com/kubernetes/kubernetes/blob/master/pkg/kubelet/kubelet.go#L3001
	return !(api.PodFailed == pod.Status.Phase || api.PodSucceeded == pod.Status.Phase)
}
