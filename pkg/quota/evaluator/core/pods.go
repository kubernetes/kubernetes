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
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/initialization"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/helper/qos"
	k8s_api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/kubeapiserver/admission/util"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/quota/generic"
)

// podResources are the set of resources managed by quota associated with pods.
var podResources = []api.ResourceName{
	api.ResourceCPU,
	api.ResourceMemory,
	api.ResourceEphemeralStorage,
	api.ResourceRequestsCPU,
	api.ResourceRequestsMemory,
	api.ResourceRequestsEphemeralStorage,
	api.ResourceLimitsCPU,
	api.ResourceLimitsMemory,
	api.ResourceLimitsEphemeralStorage,
	api.ResourcePods,
}

// listPodsByNamespaceFuncUsingClient returns a pod listing function based on the provided client.
func listPodsByNamespaceFuncUsingClient(kubeClient clientset.Interface) generic.ListFuncByNamespace {
	// TODO: ideally, we could pass dynamic client pool down into this code, and have one way of doing this.
	// unfortunately, dynamic client works with Unstructured objects, and when we calculate Usage, we require
	// structured objects.
	return func(namespace string, options metav1.ListOptions) ([]runtime.Object, error) {
		itemList, err := kubeClient.Core().Pods(namespace).List(options)
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

// NewPodEvaluator returns an evaluator that can evaluate pods
// if the specified shared informer factory is not nil, evaluator may use it to support listing functions.
func NewPodEvaluator(kubeClient clientset.Interface, f informers.SharedInformerFactory, clock clock.Clock) quota.Evaluator {
	listFuncByNamespace := listPodsByNamespaceFuncUsingClient(kubeClient)
	if f != nil {
		listFuncByNamespace = generic.ListResourceUsingInformerFunc(f, v1.SchemeGroupVersion.WithResource("pods"))
	}
	return &podEvaluator{
		listFuncByNamespace: listFuncByNamespace,
		clock:               clock,
	}
}

// podEvaluator knows how to measure usage of pods.
type podEvaluator struct {
	// knows how to list pods
	listFuncByNamespace generic.ListFuncByNamespace
	// used to track time
	clock clock.Clock
}

// Constraints verifies that all required resources are present on the pod
// In addition, it validates that the resources are valid (i.e. requests < limits)
func (p *podEvaluator) Constraints(required []api.ResourceName, item runtime.Object) error {
	pod, ok := item.(*api.Pod)
	if !ok {
		return fmt.Errorf("Unexpected input object %v", item)
	}

	// Pod level resources are often set during admission control
	// As a consequence, we want to verify that resources are valid prior
	// to ever charging quota prematurely in case they are not.
	allErrs := field.ErrorList{}
	fldPath := field.NewPath("spec").Child("containers")
	for i, ctr := range pod.Spec.Containers {
		allErrs = append(allErrs, validation.ValidateResourceRequirements(&ctr.Resources, fldPath.Index(i).Child("resources"))...)
	}
	fldPath = field.NewPath("spec").Child("initContainers")
	for i, ctr := range pod.Spec.InitContainers {
		allErrs = append(allErrs, validation.ValidateResourceRequirements(&ctr.Resources, fldPath.Index(i).Child("resources"))...)
	}
	if len(allErrs) > 0 {
		return allErrs.ToAggregate()
	}

	// TODO: fix this when we have pod level resource requirements
	// since we do not yet pod level requests/limits, we need to ensure each
	// container makes an explict request or limit for a quota tracked resource
	requiredSet := quota.ToSet(required)
	missingSet := sets.NewString()
	for i := range pod.Spec.Containers {
		enforcePodContainerConstraints(&pod.Spec.Containers[i], requiredSet, missingSet)
	}
	for i := range pod.Spec.InitContainers {
		enforcePodContainerConstraints(&pod.Spec.InitContainers[i], requiredSet, missingSet)
	}
	if len(missingSet) == 0 {
		return nil
	}
	return fmt.Errorf("must specify %s", strings.Join(missingSet.List(), ","))
}

// GroupKind that this evaluator tracks
func (p *podEvaluator) GroupKind() schema.GroupKind {
	return api.Kind("Pod")
}

// Handles returns true if the evaluator should handle the specified attributes.
func (p *podEvaluator) Handles(a admission.Attributes) bool {
	op := a.GetOperation()
	if op == admission.Create {
		return true
	}
	initializationCompletion, err := util.IsInitializationCompletion(a)
	if err != nil {
		// fail closed, will try to give an evaluation.
		utilruntime.HandleError(err)
		return true
	}
	// only uninitialized pods might be updated.
	return initializationCompletion
}

// Matches returns true if the evaluator matches the specified quota with the provided input item
func (p *podEvaluator) Matches(resourceQuota *api.ResourceQuota, item runtime.Object) (bool, error) {
	return generic.Matches(resourceQuota, item, p.MatchingResources, podMatchesScopeFunc)
}

// MatchingResources takes the input specified list of resources and returns the set of resources it matches.
func (p *podEvaluator) MatchingResources(input []api.ResourceName) []api.ResourceName {
	return quota.Intersection(input, podResources)
}

// Usage knows how to measure usage associated with pods
func (p *podEvaluator) Usage(item runtime.Object) (api.ResourceList, error) {
	// delegate to normal usage
	return PodUsageFunc(item, p.clock)
}

// UsageStats calculates aggregate usage for the object.
func (p *podEvaluator) UsageStats(options quota.UsageStatsOptions) (quota.UsageStats, error) {
	return generic.CalculateUsageStats(options, p.listFuncByNamespace, podMatchesScopeFunc, p.Usage)
}

// verifies we implement the required interface.
var _ quota.Evaluator = &podEvaluator{}

// enforcePodContainerConstraints checks for required resources that are not set on this container and
// adds them to missingSet.
func enforcePodContainerConstraints(container *api.Container, requiredSet, missingSet sets.String) {
	requests := container.Resources.Requests
	limits := container.Resources.Limits
	containerUsage := podUsageHelper(requests, limits)
	containerSet := quota.ToSet(quota.ResourceNames(containerUsage))
	if !containerSet.Equal(requiredSet) {
		difference := requiredSet.Difference(containerSet)
		missingSet.Insert(difference.List()...)
	}
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
	if request, found := requests[api.ResourceEphemeralStorage]; found {
		result[api.ResourceEphemeralStorage] = request
		result[api.ResourceRequestsEphemeralStorage] = request
	}
	if limit, found := limits[api.ResourceEphemeralStorage]; found {
		result[api.ResourceLimitsEphemeralStorage] = limit
	}
	return result
}

func toInternalPodOrError(obj runtime.Object) (*api.Pod, error) {
	pod := &api.Pod{}
	switch t := obj.(type) {
	case *v1.Pod:
		if err := k8s_api_v1.Convert_v1_Pod_To_api_Pod(t, pod, nil); err != nil {
			return nil, err
		}
	case *api.Pod:
		pod = t
	default:
		return nil, fmt.Errorf("expect *api.Pod or *v1.Pod, got %v", t)
	}
	return pod, nil
}

// podMatchesScopeFunc is a function that knows how to evaluate if a pod matches a scope
func podMatchesScopeFunc(scope api.ResourceQuotaScope, object runtime.Object) (bool, error) {
	pod, err := toInternalPodOrError(object)
	if err != nil {
		return false, err
	}
	switch scope {
	case api.ResourceQuotaScopeTerminating:
		return isTerminating(pod), nil
	case api.ResourceQuotaScopeNotTerminating:
		return !isTerminating(pod), nil
	case api.ResourceQuotaScopeBestEffort:
		return isBestEffort(pod), nil
	case api.ResourceQuotaScopeNotBestEffort:
		return !isBestEffort(pod), nil
	}
	return false, nil
}

// PodUsageFunc returns the quota usage for a pod.
// A pod is charged for quota if the following are not true.
//  - pod has a terminal phase (failed or succeeded)
//  - pod has been marked for deletion and grace period has expired
func PodUsageFunc(obj runtime.Object, clock clock.Clock) (api.ResourceList, error) {
	pod, err := toInternalPodOrError(obj)
	if err != nil {
		return api.ResourceList{}, err
	}
	// by convention, we do not quota pods that have reached end-of life
	if !QuotaPod(pod, clock) {
		return api.ResourceList{}, nil
	}
	// Only charge pod count for uninitialized pod.
	if utilfeature.DefaultFeatureGate.Enabled(features.Initializers) {
		if !initialization.IsInitialized(pod.Initializers) {
			result := api.ResourceList{}
			result[api.ResourcePods] = resource.MustParse("1")
			return result, nil
		}
	}
	requests := api.ResourceList{}
	limits := api.ResourceList{}
	// TODO: ideally, we have pod level requests and limits in the future.
	for i := range pod.Spec.Containers {
		requests = quota.Add(requests, pod.Spec.Containers[i].Resources.Requests)
		limits = quota.Add(limits, pod.Spec.Containers[i].Resources.Limits)
	}
	// InitContainers are run sequentially before other containers start, so the highest
	// init container resource is compared against the sum of app containers to determine
	// the effective usage for both requests and limits.
	for i := range pod.Spec.InitContainers {
		requests = quota.Max(requests, pod.Spec.InitContainers[i].Resources.Requests)
		limits = quota.Max(limits, pod.Spec.InitContainers[i].Resources.Limits)
	}

	return podUsageHelper(requests, limits), nil
}

func isBestEffort(pod *api.Pod) bool {
	return qos.GetPodQOS(pod) == api.PodQOSBestEffort
}

func isTerminating(pod *api.Pod) bool {
	if pod.Spec.ActiveDeadlineSeconds != nil && *pod.Spec.ActiveDeadlineSeconds >= int64(0) {
		return true
	}
	return false
}

// QuotaPod returns true if the pod is eligible to track against a quota
// A pod is eligible for quota, unless any of the following are true:
//  - pod has a terminal phase (failed or succeeded)
//  - pod has been marked for deletion and grace period has expired.
func QuotaPod(pod *api.Pod, clock clock.Clock) bool {
	// if pod is terminal, ignore it for quota
	if api.PodFailed == pod.Status.Phase || api.PodSucceeded == pod.Status.Phase {
		return false
	}
	// deleted pods that should be gone should not be charged to user quota.
	// this can happen if a node is lost, and the kubelet is never able to confirm deletion.
	// even though the cluster may have drifting clocks, quota makes a reasonable effort
	// to balance cluster needs against user needs.  user's do not control clocks,
	// but at worst a small drive in clocks will only slightly impact quota.
	if pod.DeletionTimestamp != nil && pod.DeletionGracePeriodSeconds != nil {
		now := clock.Now()
		deletionTime := pod.DeletionTimestamp.Time
		gracePeriod := time.Duration(*pod.DeletionGracePeriodSeconds) * time.Second
		if now.After(deletionTime.Add(gracePeriod)) {
			return false
		}
	}
	return true
}

// QuotaV1Pod returns true if the pod is eligible to track against a quota
// if it's not in a terminal state according to its phase.
func QuotaV1Pod(pod *v1.Pod, clock clock.Clock) bool {
	// if pod is terminal, ignore it for quota
	if v1.PodFailed == pod.Status.Phase || v1.PodSucceeded == pod.Status.Phase {
		return false
	}
	// if pods are stuck terminating (for example, a node is lost), we do not want
	// to charge the user for that pod in quota because it could prevent them from
	// scaling up new pods to service their application.
	if pod.DeletionTimestamp != nil && pod.DeletionGracePeriodSeconds != nil {
		now := clock.Now()
		deletionTime := pod.DeletionTimestamp.Time
		gracePeriod := time.Duration(*pod.DeletionGracePeriodSeconds) * time.Second
		if now.After(deletionTime.Add(gracePeriod)) {
			return false
		}
	}
	return true
}
