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

	corev1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	quota "k8s.io/apiserver/pkg/quota/v1"
	"k8s.io/apiserver/pkg/quota/v1/generic"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/utils/clock"

	resourcehelper "k8s.io/component-helpers/resource"
	api "k8s.io/kubernetes/pkg/apis/core"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/features"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// the name used for object count quota
var podObjectCountName = generic.ObjectCountQuotaResourceNameFor(corev1.SchemeGroupVersion.WithResource("pods").GroupResource())
var resourceRequestsDeviceClassPrefix = corev1.DefaultResourceRequestsPrefix + resourceapi.ResourceDeviceClassPrefix

// podResources are the set of resources managed by quota associated with pods.
var podResources = []corev1.ResourceName{
	podObjectCountName,
	corev1.ResourceCPU,
	corev1.ResourceMemory,
	corev1.ResourceEphemeralStorage,
	corev1.ResourceRequestsCPU,
	corev1.ResourceRequestsMemory,
	corev1.ResourceRequestsEphemeralStorage,
	corev1.ResourceLimitsCPU,
	corev1.ResourceLimitsMemory,
	corev1.ResourceLimitsEphemeralStorage,
	corev1.ResourcePods,
}

// podResourcePrefixes are the set of prefixes for resources (Hugepages, and other
// potential extended resources with specific prefix) managed by quota associated with pods.
var podResourcePrefixes = []string{
	corev1.ResourceHugePagesPrefix,
	corev1.ResourceRequestsHugePagesPrefix,
}

// requestedResourcePrefixes are the set of prefixes for resources
// that might be declared in pod's Resources.Requests/Limits
var requestedResourcePrefixes = []string{
	corev1.ResourceHugePagesPrefix,
}

// maskResourceWithPrefix mask resource with certain prefix
// e.g. hugepages-XXX -> requests.hugepages-XXX
func maskResourceWithPrefix(resource corev1.ResourceName, prefix string) corev1.ResourceName {
	return corev1.ResourceName(fmt.Sprintf("%s%s", prefix, string(resource)))
}

// isExtendedResourceNameForQuota returns true if the extended resource name
// has the quota related resource prefix.
func isExtendedResourceNameForQuota(name corev1.ResourceName) bool {
	// As overcommit is not supported by extended resources for now,
	// quota objects in format of "requests.resourceName" is allowed
	nonNative := !helper.IsNativeResource(name)
	// allow the implicit extended resource name in the format of
	// requests.deviceclass.resource.kubernetes.io/device-class-name
	implicitExtendedResource := strings.HasPrefix(string(name), resourceRequestsDeviceClassPrefix)
	// name starts with 'requests.'
	isQuotaRequest := strings.HasPrefix(string(name), corev1.DefaultResourceRequestsPrefix)

	return (nonNative || implicitExtendedResource) && isQuotaRequest
}

// NOTE: it was a mistake, but if a quota tracks cpu or memory related resources,
// the incoming pod is required to have those values set.  we should not repeat
// this mistake for other future resources (gpus, ephemeral-storage,etc).
// do not add more resources to this list!
var validationSet = sets.NewString(
	string(corev1.ResourceCPU),
	string(corev1.ResourceMemory),
	string(corev1.ResourceRequestsCPU),
	string(corev1.ResourceRequestsMemory),
	string(corev1.ResourceLimitsCPU),
	string(corev1.ResourceLimitsMemory),
)

// NewPodEvaluator returns an evaluator that can evaluate pods
func NewPodEvaluator(f quota.ListerForResourceFunc, clock clock.Clock) quota.Evaluator {
	listFuncByNamespace := generic.ListResourceUsingListerFunc(f, corev1.SchemeGroupVersion.WithResource("pods"))
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
func (p *podEvaluator) Constraints(required []corev1.ResourceName, item runtime.Object) error {
	pod, err := toExternalPodOrError(item)
	if err != nil {
		return err
	}

	// As mentioned in the subsequent comment, the older versions required explicit
	// resource requests for CPU & memory for each container if resource quotas were
	// enabled for these resources. This was a design flaw as resource validation is
	// coupled with quota enforcement. With pod-level resources
	// feature, container-level resources are not mandatory. Hence the check for
	// missing container requests, for CPU/memory resources that have quotas set,
	// is skipped when pod-level resources feature is enabled and resources are set
	// at pod level.
	if feature.DefaultFeatureGate.Enabled(features.PodLevelResources) && resourcehelper.IsPodLevelResourcesSet(pod) {
		return nil
	}

	// BACKWARD COMPATIBILITY REQUIREMENT: if we quota cpu or memory, then each container
	// must make an explicit request for the resource.  this was a mistake.  it coupled
	// validation with resource counting, but we did this before QoS was even defined.
	// let's not make that mistake again with other resources now that QoS is defined.
	requiredSet := quota.ToSet(required).Intersection(validationSet)
	missingSetResourceToContainerNames := make(map[string]sets.String)
	for i := range pod.Spec.Containers {
		enforcePodContainerConstraints(&pod.Spec.Containers[i], requiredSet, missingSetResourceToContainerNames)
	}
	for i := range pod.Spec.InitContainers {
		enforcePodContainerConstraints(&pod.Spec.InitContainers[i], requiredSet, missingSetResourceToContainerNames)
	}
	if len(missingSetResourceToContainerNames) == 0 {
		return nil
	}
	var resources = sets.NewString()
	for resource := range missingSetResourceToContainerNames {
		resources.Insert(resource)
	}
	var errorMessages = make([]string, 0, len(missingSetResourceToContainerNames))
	for _, resource := range resources.List() {
		errorMessages = append(errorMessages, fmt.Sprintf("%s for: %s", resource, strings.Join(missingSetResourceToContainerNames[resource].List(), ",")))
	}
	return fmt.Errorf("must specify %s", strings.Join(errorMessages, "; "))
}

// GroupResource that this evaluator tracks
func (p *podEvaluator) GroupResource() schema.GroupResource {
	return corev1.SchemeGroupVersion.WithResource("pods").GroupResource()
}

// Handles returns true if the evaluator should handle the specified attributes.
func (p *podEvaluator) Handles(a admission.Attributes) bool {
	op := a.GetOperation()
	switch a.GetSubresource() {
	case "":
		if op == admission.Update {
			pod, err1 := toExternalPodOrError(a.GetObject())
			oldPod, err2 := toExternalPodOrError(a.GetOldObject())
			if err1 != nil || err2 != nil {
				return false
			}
			// when scope changed
			if IsTerminating(oldPod) != IsTerminating(pod) {
				return true
			}
		}
		return op == admission.Create
	case "resize":
		return op == admission.Update
	default:
		return false
	}
}

// Matches returns true if the evaluator matches the specified quota with the provided input item
func (p *podEvaluator) Matches(resourceQuota *corev1.ResourceQuota, item runtime.Object) (bool, error) {
	return generic.Matches(resourceQuota, item, p.MatchingResources, podMatchesScopeFunc)
}

// MatchingResources takes the input specified list of resources and returns the set of resources it matches.
func (p *podEvaluator) MatchingResources(input []corev1.ResourceName) []corev1.ResourceName {
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
func (p *podEvaluator) MatchingScopes(item runtime.Object, scopeSelectors []corev1.ScopedResourceSelectorRequirement) ([]corev1.ScopedResourceSelectorRequirement, error) {
	matchedScopes := []corev1.ScopedResourceSelectorRequirement{}
	for _, selector := range scopeSelectors {
		match, err := podMatchesScopeFunc(selector, item)
		if err != nil {
			return []corev1.ScopedResourceSelectorRequirement{}, fmt.Errorf("error on matching scope %v: %v", selector, err)
		}
		if match {
			matchedScopes = append(matchedScopes, selector)
		}
	}
	return matchedScopes, nil
}

// UncoveredQuotaScopes takes the input matched scopes which are limited by configuration and the matched quota scopes.
// It returns the scopes which are in limited scopes but don't have a corresponding covering quota scope
func (p *podEvaluator) UncoveredQuotaScopes(limitedScopes []corev1.ScopedResourceSelectorRequirement, matchedQuotaScopes []corev1.ScopedResourceSelectorRequirement) ([]corev1.ScopedResourceSelectorRequirement, error) {
	uncoveredScopes := []corev1.ScopedResourceSelectorRequirement{}
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
func (p *podEvaluator) Usage(item runtime.Object) (corev1.ResourceList, error) {
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
func enforcePodContainerConstraints(container *corev1.Container, requiredSet sets.String, missingSetResourceToContainerNames map[string]sets.String) {
	requests := container.Resources.Requests
	limits := container.Resources.Limits
	containerUsage := podComputeUsageHelper(requests, limits)
	containerSet := quota.ToSet(quota.ResourceNames(containerUsage))
	if !containerSet.Equal(requiredSet) {
		if difference := requiredSet.Difference(containerSet); difference.Len() != 0 {
			for _, diff := range difference.List() {
				if _, ok := missingSetResourceToContainerNames[diff]; !ok {
					missingSetResourceToContainerNames[diff] = sets.NewString(container.Name)
				} else {
					missingSetResourceToContainerNames[diff].Insert(container.Name)
				}
			}
		}
	}
}

// podComputeUsageHelper can summarize the pod compute quota usage based on requests and limits
func podComputeUsageHelper(requests corev1.ResourceList, limits corev1.ResourceList) corev1.ResourceList {
	result := corev1.ResourceList{}
	result[corev1.ResourcePods] = resource.MustParse("1")
	if request, found := requests[corev1.ResourceCPU]; found {
		result[corev1.ResourceCPU] = request
		result[corev1.ResourceRequestsCPU] = request
	}
	if limit, found := limits[corev1.ResourceCPU]; found {
		result[corev1.ResourceLimitsCPU] = limit
	}
	if request, found := requests[corev1.ResourceMemory]; found {
		result[corev1.ResourceMemory] = request
		result[corev1.ResourceRequestsMemory] = request
	}
	if limit, found := limits[corev1.ResourceMemory]; found {
		result[corev1.ResourceLimitsMemory] = limit
	}
	if request, found := requests[corev1.ResourceEphemeralStorage]; found {
		result[corev1.ResourceEphemeralStorage] = request
		result[corev1.ResourceRequestsEphemeralStorage] = request
	}
	if limit, found := limits[corev1.ResourceEphemeralStorage]; found {
		result[corev1.ResourceLimitsEphemeralStorage] = limit
	}
	for resource, request := range requests {
		// for resources with certain prefix, e.g. hugepages
		if quota.ContainsPrefix(requestedResourcePrefixes, resource) {
			result[resource] = request
			result[maskResourceWithPrefix(resource, corev1.DefaultResourceRequestsPrefix)] = request
		}
		// for extended resources
		if schedutil.IsDRAExtendedResourceName(resource) {
			// only quota objects in format of "requests.resourceName" is allowed for extended resource.
			result[maskResourceWithPrefix(resource, corev1.DefaultResourceRequestsPrefix)] = request
		}
	}

	return result
}

func toExternalPodOrError(obj runtime.Object) (*corev1.Pod, error) {
	pod := &corev1.Pod{}
	switch t := obj.(type) {
	case *corev1.Pod:
		pod = t
	case *api.Pod:
		if err := k8s_api_v1.Convert_core_Pod_To_v1_Pod(t, pod, nil); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("expect *api.Pod or *v1.Pod, got %v", t)
	}
	return pod, nil
}

// podMatchesScopeFunc is a function that knows how to evaluate if a pod matches a scope
func podMatchesScopeFunc(selector corev1.ScopedResourceSelectorRequirement, object runtime.Object) (bool, error) {
	pod, err := toExternalPodOrError(object)
	if err != nil {
		return false, err
	}
	switch selector.ScopeName {
	case corev1.ResourceQuotaScopeTerminating:
		return IsTerminating(pod), nil
	case corev1.ResourceQuotaScopeNotTerminating:
		return !IsTerminating(pod), nil
	case corev1.ResourceQuotaScopeBestEffort:
		return isBestEffort(pod), nil
	case corev1.ResourceQuotaScopeNotBestEffort:
		return !isBestEffort(pod), nil
	case corev1.ResourceQuotaScopePriorityClass:
		if selector.Operator == corev1.ScopeSelectorOpExists {
			// This is just checking for existence of a priorityClass on the pod,
			// no need to take the overhead of selector parsing/evaluation.
			return len(pod.Spec.PriorityClassName) != 0, nil
		}
		return podMatchesSelector(pod, selector)
	case corev1.ResourceQuotaScopeCrossNamespacePodAffinity:
		return usesCrossNamespacePodAffinity(pod), nil
	}
	return false, nil
}

// PodUsageFunc returns the quota usage for a pod.
// A pod is charged for quota if the following are not true.
//   - pod has a terminal phase (failed or succeeded)
//   - pod has been marked for deletion and grace period has expired
func PodUsageFunc(obj runtime.Object, clock clock.Clock) (corev1.ResourceList, error) {
	pod, err := toExternalPodOrError(obj)
	if err != nil {
		return corev1.ResourceList{}, err
	}

	// always quota the object count (even if the pod is end of life)
	// object count quotas track all objects that are in storage.
	// where "pods" tracks all pods that have not reached a terminal state,
	// count/pods tracks all pods independent of state.
	result := corev1.ResourceList{
		podObjectCountName: *(resource.NewQuantity(1, resource.DecimalSI)),
	}

	// by convention, we do not quota compute resources that have reached end-of life
	// note: the "pods" resource is considered a compute resource since it is tied to life-cycle.
	if !QuotaV1Pod(pod, clock) {
		return result, nil
	}

	opts := resourcehelper.PodResourcesOptions{
		UseStatusResources: feature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling),
		// SkipPodLevelResources is set to false when PodLevelResources feature is enabled.
		SkipPodLevelResources: !feature.DefaultFeatureGate.Enabled(features.PodLevelResources),
	}
	requests := resourcehelper.PodRequests(pod, opts)
	limits := resourcehelper.PodLimits(pod, opts)

	result = quota.Add(result, podComputeUsageHelper(requests, limits))
	return result, nil
}

func isBestEffort(pod *corev1.Pod) bool {
	return qos.GetPodQOS(pod) == corev1.PodQOSBestEffort
}

func IsTerminating(pod *corev1.Pod) bool {
	if pod.Spec.ActiveDeadlineSeconds != nil && *pod.Spec.ActiveDeadlineSeconds >= int64(0) {
		return true
	}
	return false
}

func podMatchesSelector(pod *corev1.Pod, selector corev1.ScopedResourceSelectorRequirement) (bool, error) {
	labelSelector, err := helper.ScopedResourceSelectorRequirementsAsSelector(selector)
	if err != nil {
		return false, fmt.Errorf("failed to parse and convert selector: %v", err)
	}
	var m map[string]string
	if len(pod.Spec.PriorityClassName) != 0 {
		m = map[string]string{string(corev1.ResourceQuotaScopePriorityClass): pod.Spec.PriorityClassName}
	}
	if labelSelector.Matches(labels.Set(m)) {
		return true, nil
	}
	return false, nil
}

func crossNamespacePodAffinityTerm(term *corev1.PodAffinityTerm) bool {
	return len(term.Namespaces) != 0 || term.NamespaceSelector != nil
}

func crossNamespacePodAffinityTerms(terms []corev1.PodAffinityTerm) bool {
	for _, t := range terms {
		if crossNamespacePodAffinityTerm(&t) {
			return true
		}
	}
	return false
}

func crossNamespaceWeightedPodAffinityTerms(terms []corev1.WeightedPodAffinityTerm) bool {
	for _, t := range terms {
		if crossNamespacePodAffinityTerm(&t.PodAffinityTerm) {
			return true
		}
	}
	return false
}

func usesCrossNamespacePodAffinity(pod *corev1.Pod) bool {
	if pod == nil || pod.Spec.Affinity == nil {
		return false
	}

	affinity := pod.Spec.Affinity.PodAffinity
	if affinity != nil {
		if crossNamespacePodAffinityTerms(affinity.RequiredDuringSchedulingIgnoredDuringExecution) {
			return true
		}
		if crossNamespaceWeightedPodAffinityTerms(affinity.PreferredDuringSchedulingIgnoredDuringExecution) {
			return true
		}
	}

	antiAffinity := pod.Spec.Affinity.PodAntiAffinity
	if antiAffinity != nil {
		if crossNamespacePodAffinityTerms(antiAffinity.RequiredDuringSchedulingIgnoredDuringExecution) {
			return true
		}
		if crossNamespaceWeightedPodAffinityTerms(antiAffinity.PreferredDuringSchedulingIgnoredDuringExecution) {
			return true
		}
	}

	return false
}

// QuotaV1Pod returns true if the pod is eligible to track against a quota
// if it's not in a terminal state according to its phase.
func QuotaV1Pod(pod *corev1.Pod, clock clock.Clock) bool {
	// if pod is terminal, ignore it for quota
	if corev1.PodFailed == pod.Status.Phase || corev1.PodSucceeded == pod.Status.Phase {
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
