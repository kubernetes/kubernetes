/*
Copyright 2025 The Kubernetes Authors.

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

package podautoscaler

import (
	"fmt"

	autoscalingv2 "k8s.io/api/autoscaling/v2"
	v1 "k8s.io/api/core/v1"

	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	appsv1client "k8s.io/client-go/kubernetes/typed/apps/v1"
)

type FilterOptions struct {
	ScaleTargetRef *autoscalingv2.CrossVersionObjectReference
}

// NewPodFilter creates a new PodFilter with the given strategy and options
func NewPodFilter(strategyName string, opts FilterOptions) PodFilter {
	if strategyName == string(autoscalingv2.OwnerReferences) {
		return &OwnerReferencesFilter{
			filterOptions: opts,
		}
	}
	// default to LabelSelector
	return &LabelSelectorFilter{
		filterOptions: opts,
	}
}

// PodFilter defines an interface for filtering pods based on various strategies
type PodFilter interface {
	// Filter returns the subset of pods that should be considered for metrics calculation,
	// along with the pods that were filtered out
	Filter(pods []*v1.Pod) (filtered []*v1.Pod, unfiltered []*v1.Pod, err error)
	// Name returns the name of the filter strategy for logging purposes
	Name() string
	WithRESTMapper(mapper apimeta.RESTMapper) PodFilter
	WithCache(cache *ControllerCache) PodFilter
	WithDynamicClient(client dynamic.Interface) PodFilter
}

// OwnerReferencesFilter filters pods by ownership chain
type OwnerReferencesFilter struct {
	filterOptions FilterOptions
	Client        appsv1client.AppsV1Interface
	RESTMapper    apimeta.RESTMapper
	Cache         *ControllerCache
	dynamicClient dynamic.Interface
}

func (f *OwnerReferencesFilter) WithClient(client appsv1client.AppsV1Interface) PodFilter {
	f.Client = client
	return f
}

func (f *OwnerReferencesFilter) WithRESTMapper(mapper apimeta.RESTMapper) PodFilter {
	f.RESTMapper = mapper
	return f
}

func (f *OwnerReferencesFilter) Name() string {
	return string(autoscalingv2.OwnerReferences)
}

func (f *OwnerReferencesFilter) WithCache(cache *ControllerCache) PodFilter {
	f.Cache = cache
	return f
}

func (f *OwnerReferencesFilter) WithDynamicClient(client dynamic.Interface) PodFilter {
	f.dynamicClient = client
	return f
}

func (f *OwnerReferencesFilter) Filter(pods []*v1.Pod) ([]*v1.Pod, []*v1.Pod, error) {
	if f.Cache == nil {
		return nil, nil, fmt.Errorf("cache is required for OwnerReferencesFilter") // TODO: how to handle this?
	}

	if f.filterOptions.ScaleTargetRef == nil {
		return nil, nil, fmt.Errorf("ScaleTargetRef is required for OwnerReferencesFilter")
	}

	// If no pods to filter, return empty slice
	if len(pods) == 0 {
		return pods, nil, nil
	}

	filteredPods := make([]*v1.Pod, 0, len(pods))
	unfilteredPods := make([]*v1.Pod, 0, len(pods))

	namespace := pods[0].Namespace
	targetRef := f.filterOptions.ScaleTargetRef

	// Check ownership for each pod
	for _, pod := range pods {
		isOwned, err := f.isPodOwnedByTarget(pod, *targetRef, namespace)
		if err != nil {
			// On error, assume pod is not owned
			// TODO: is this how to handle this?
			unfilteredPods = append(unfilteredPods, pod)
			continue
		}

		if isOwned {
			filteredPods = append(filteredPods, pod)
		} else {
			unfilteredPods = append(unfilteredPods, pod)
		}
	}

	return filteredPods, unfilteredPods, nil
}

func (f *OwnerReferencesFilter) isPodOwnedByTarget(pod *v1.Pod, targetRef autoscalingv2.CrossVersionObjectReference, namespace string) (bool, error) {
	// TODO(omerap12): Add a cache to remember ownership results for each HPA check.
	// This would help when many pods have the same owners, so we don't need
	// to check the same ownership path multiple times.
	const maxOwnershipChainLength = 10

	// Use BFS to traverse all possible ownership paths
	currentLevel := []*unstructured.Unstructured{}

	// Start with the pod itself
	podObj := &unstructured.Unstructured{}
	podObj.SetName(pod.Name)
	podObj.SetNamespace(pod.Namespace)
	podObj.SetUID(pod.UID)
	podObj.SetOwnerReferences(pod.OwnerReferences)
	currentLevel = append(currentLevel, podObj)

	// Track visited resources to prevent cycles
	visited := make(map[string]bool)

	for depth := 0; depth < maxOwnershipChainLength && len(currentLevel) > 0; depth++ {
		nextLevel := []*unstructured.Unstructured{}

		for _, current := range currentLevel {
			// Create unique key for cycle detection
			key := fmt.Sprintf("%s/%s", current.GetNamespace(), current.GetName())
			if visited[key] {
				continue // Skip already visited resources
			}
			visited[key] = true

			// Check if current object is our target
			isTarget := f.isTargetMatch(current, targetRef)
			if isTarget {
				return true, nil
			}

			// Process all owner references for this object
			ownerRefs := current.GetOwnerReferences()
			for _, ownerRef := range ownerRefs {

				// Check if this owner directly matches our target
				isOwnerMatch := f.isOwnerRefMatch(ownerRef, targetRef)

				if isOwnerMatch {
					return true, nil
				}

				// Fetch the owner for next level traversal
				owner, err := f.Cache.GetResource(namespace, ownerRef)
				if err != nil {
					// Skip owners we can't fetch, but continue with others
					continue
				}
				// Add to next level for BFS traversal
				nextLevel = append(nextLevel, owner)
			}
		}

		// Move to next level
		currentLevel = nextLevel
	}

	// Check if we exceeded max depth with remaining objects to process
	if len(currentLevel) > 0 {
		return false, fmt.Errorf("maximum ownership chain depth (%d) exceeded", maxOwnershipChainLength)
	}
	return false, nil
}

// isTargetMatch checks if the current object matches the target reference
func (f *OwnerReferencesFilter) isTargetMatch(obj *unstructured.Unstructured, targetRef autoscalingv2.CrossVersionObjectReference) bool {
	if obj.GetName() != targetRef.Name {
		return false
	}

	// Get the object's GVK
	gvk := obj.GroupVersionKind()

	// Parse target's API version
	targetGV, err := schema.ParseGroupVersion(targetRef.APIVersion)
	if err != nil {
		return false
	}

	// Compare Kind and API Version
	return gvk.Kind == targetRef.Kind && gvk.Group == targetGV.Group && gvk.Version == targetGV.Version
}

// isOwnerRefMatch checks if an owner reference matches the target reference
func (f *OwnerReferencesFilter) isOwnerRefMatch(ownerRef metav1.OwnerReference, targetRef autoscalingv2.CrossVersionObjectReference) bool {
	kindMatch := ownerRef.Kind == targetRef.Kind
	nameMatch := ownerRef.Name == targetRef.Name
	apiVersionMatch := ownerRef.APIVersion == targetRef.APIVersion

	result := kindMatch && nameMatch && apiVersionMatch
	return result
}

// LabelSelectorFilter uses the default label selector strategy
type LabelSelectorFilter struct {
	filterOptions FilterOptions
}

// The default behavior - keep all pods
func (f *LabelSelectorFilter) Filter(pods []*v1.Pod) ([]*v1.Pod, []*v1.Pod, error) {
	return pods, []*v1.Pod{}, nil
}

func (f *LabelSelectorFilter) Name() string {
	return string(autoscalingv2.LabelSelector)
}

func (f *LabelSelectorFilter) WithDynamicClient(client dynamic.Interface) PodFilter {
	// No-op for label selector
	return f
}

func (f *LabelSelectorFilter) WithRESTMapper(mapper apimeta.RESTMapper) PodFilter {
	// No-op for label selector
	return f
}

func (f *LabelSelectorFilter) WithCache(cache *ControllerCache) PodFilter {
	// No-op for label selector
	return f
}
