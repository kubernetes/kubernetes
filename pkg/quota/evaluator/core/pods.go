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
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"k8s.io/apimachinery/pkg/util/clock"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	"k8s.io/kubernetes/pkg/apis/core/helper/qos"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/kubeapiserver/admission/util"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/quota/generic"
)

// the name used for object count quota
var podObjectCountName = generic.ObjectCountQuotaResourceNameFor(v1.SchemeGroupVersion.WithResource("pods").GroupResource())

// podResources are the set of resources managed by quota associated with pods.
var podResources = []api.ResourceName{
	podObjectCountName,
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

// podResourcePrefixes are the set of prefixes for resources (Hugepages, and other
// potential extended reources with specific prefix) managed by quota associated with pods.
var podResourcePrefixes = []string{
	api.ResourceHugePagesPrefix,
	api.ResourceRequestsHugePagesPrefix,
}

// requestedResourcePrefixes are the set of prefixes for resources
// that might be declared in pod's Resources.Requests/Limits
var requestedResourcePrefixes = []string{
	api.ResourceHugePagesPrefix,
}

// maskResourceWithPrefix mask resource with certain prefix
// e.g. hugepages-XXX -> requests.hugepages-XXX
func maskResourceWithPrefix(resource api.ResourceName, prefix string) api.ResourceName {
	return api.ResourceName(fmt.Sprintf("%s%s", prefix, string(resource)))
}

// isExtendedResourceNameForQuota returns true if the extended resource name
// has the quota related resource prefix.
func isExtendedResourceNameForQuota(name api.ResourceName) bool {
	// As overcommit is not supported by extended resources for now,
	// only quota objects in format of "requests.resourceName" is allowed.
	return !helper.IsNativeResource(name) && strings.HasPrefix(string(name), api.DefaultResourceRequestsPrefix)
}

// NOTE: it was a mistake, but if a quota tracks cpu or memory related resources,
// the incoming pod is required to have those values set.  we should not repeat
// this mistake for other future resources (gpus, ephemeral-storage,etc).
// do not add more resources to this list!
var validationSet = sets.NewString(
	string(api.ResourceCPU),
	string(api.ResourceMemory),
	string(api.ResourceRequestsCPU),
	string(api.ResourceRequestsMemory),
	string(api.ResourceLimitsCPU),
	string(api.ResourceLimitsMemory),
)

// NewPodEvaluator returns an evaluator that can evaluate pods
func NewPodEvaluator(f quota.ListerForResourceFunc, clock clock.Clock) quota.Evaluator {
	listFuncByNamespace := generic.ListResourceUsingListerFunc(f, v1.SchemeGroupVersion.WithResource("pods"))
	podEvaluator := &podEvaluator{listFuncByNamespace: listFuncByNamespace, clock: clock}
	return podEvaluator
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
		return fmt.Errorf("unexpected input object %v", item)
	}

	// BACKWARD COMPATIBILITY REQUIREMENT: if we quota cpu or memory, then each container
	// must make an explicit request for the resource.  this was a mistake.  it coupled
	// validation with resource counting, but we did this before QoS was even defined.
	// let's not make that mistake again with other resources now that QoS is defined.
	requiredSet := quota.ToSet(required).Intersection(validationSet)
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

// GroupResource that this evaluator tracks
func (p *podEvaluator) GroupResource() schema.GroupResource {
	return v1.SchemeGroupVersion.WithResource("pods").GroupResource()
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
	result := quota.Intersection(input, podResources)
	for _, resource := range input {
		// for resources with certain prefix, e.g. hugepages
		if quota.ContainsPrefix(podResourcePrefixes, resource) {
			result = append(result, resource)
		}
		// for extended resources
		if isExtendedResourceNameForQuota(resource) {
			result = append(result, resource)
		}
	}

	return result
}

// MatchingScopes takes the input specified list of scopes and pod object. Returns the set of scope selectors pod matches.
func (p *podEvaluator) MatchingScopes(item runtime.Object, scopeSelectors []api.ScopedResourceSelectorRequirement) ([]api.ScopedResourceSelectorRequirement, error) {
	matchedScopes := []api.ScopedResourceSelectorRequirement{}
	for _, selector := range scopeSelectors {
		match, err := podMatchesScopeFunc(selector, item)
		if err != nil {
			return []api.ScopedResourceSelectorRequirement{}, fmt.Errorf("error on matching scope %v: %v", selector, err)
		}
		if match {
			matchedScopes = append(matchedScopes, selector)
		}
	}
	return matchedScopes, nil
}

// UncoveredQuotaScopes takes the input matched scopes which are limited by configuration and the matched quota scopes.
// It returns the scopes which are in limited scopes but dont have a corresponding covering quota scope
func (p *podEvaluator) UncoveredQuotaScopes(limitedScopes []api.ScopedResourceSelectorRequirement, matchedQuotaScopes []api.ScopedResourceSelectorRequirement) ([]api.ScopedResourceSelectorRequirement, error) {
	uncoveredScopes := []api.ScopedResourceSelectorRequirement{}
	for _, selector := range limitedScopes {
		isCovered := false
		for _, matchedScopeSelector := range matchedQuotaScopes {
			if matchedScopeSelector.ScopeName == selector.ScopeName {
				isCovered = true
				break
			}
		}

		if !isCovered {
			uncoveredScopes = append(uncoveredScopes, selector)
		}
	}
	return uncoveredScopes, nil
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
	containerUsage := podComputeUsageHelper(requests, limits)
	containerSet := quota.ToSet(quota.ResourceNames(containerUsage))
	if !containerSet.Equal(requiredSet) {
		difference := requiredSet.Difference(containerSet)
		missingSet.Insert(difference.List()...)
	}
}

// podComputeUsageHelper can summarize the pod compute quota usage based on requests and limits
func podComputeUsageHelper(requests api.ResourceList, limits api.ResourceList) api.ResourceList {
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
	for resource, request := range requests {
		// for resources with certain prefix, e.g. hugepages
		if quota.ContainsPrefix(requestedResourcePrefixes, resource) {
			result[resource] = request
			result[maskResourceWithPrefix(resource, api.DefaultResourceRequestsPrefix)] = request
		}
		// for extended resources
		if helper.IsExtendedResourceName(resource) {
			// only quota objects in format of "requests.resourceName" is allowed for extended resource.
			result[maskResourceWithPrefix(resource, api.DefaultResourceRequestsPrefix)] = request
		}
	}

	return result
}

func toInternalPodOrError(obj runtime.Object) (*api.Pod, error) {
	pod := &api.Pod{}
	switch t := obj.(type) {
	case *v1.Pod:
		if err := k8s_api_v1.Convert_v1_Pod_To_core_Pod(t, pod, nil); err != nil {
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
func podMatchesScopeFunc(selector api.ScopedResourceSelectorRequirement, object runtime.Object) (bool, error) {
	pod, err := toInternalPodOrError(object)
	if err != nil {
		return false, err
	}
	switch selector.ScopeName {
	case api.ResourceQuotaScopeTerminating:
		return isTerminating(pod), nil
	case api.ResourceQuotaScopeNotTerminating:
		return !isTerminating(pod), nil
	case api.ResourceQuotaScopeBestEffort:
		return isBestEffort(pod), nil
	case api.ResourceQuotaScopeNotBestEffort:
		return !isBestEffort(pod), nil
	case api.ResourceQuotaScopePriorityClass:
		return podMatchesSelector(pod, selector)
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

	// always quota the object count (even if the pod is end of life)
	// object count quotas track all objects that are in storage.
	// where "pods" tracks all pods that have not reached a terminal state,
	// count/pods tracks all pods independent of state.
	result := api.ResourceList{
		podObjectCountName: *(resource.NewQuantity(1, resource.DecimalSI)),
	}

	// by convention, we do not quota compute resources that have reached end-of life
	// note: the "pods" resource is considered a compute resource since it is tied to life-cycle.
	if !QuotaPod(pod, clock) {
		return result, nil
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

	result = quota.Add(result, podComputeUsageHelper(requests, limits))
	return result, nil
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

func podMatchesSelector(pod *api.Pod, selector api.ScopedResourceSelectorRequirement) (bool, error) {
	labelSelector, err := helper.ScopedResourceSelectorRequirementsAsSelector(selector)
	if err != nil {
		return false, fmt.Errorf("failed to parse and convert selector: %v", err)
	}
	var m map[string]string
	if len(pod.Spec.PriorityClassName) != 0 {
		m = map[string]string{string(api.ResourceQuotaScopePriorityClass): pod.Spec.PriorityClassName}
	}
	if labelSelector.Matches(labels.Set(m)) {
		return true, nil
	}
	return false, nil
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
